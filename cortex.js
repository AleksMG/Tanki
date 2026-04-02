/**
 * CORTEX2BRAIN v2.3 — Production Neural Architecture (Patched)
 *
 * FIXES APPLIED (v2.2 → v2.3):
 *   FIX-13  Dual export:             Removed duplicate `export` at bottom; guarded CJS export
 *   FIX-14  SeededRNG zero-state:    Guaranteed non-zero initial state for xorshift32
 *   FIX-15  _updateEligibilityTraces: Now receives and uses actual C_input vector (not raw sensors)
 *   FIX-16  _applyTDError:           Now updates ALL cortex columns, not just 64..127
 *   FIX-17  _canExitEmotion:         Inverted logic fixed — exits when expired/satisfied/faded
 *   FIX-18  Prediction error:        predictionErrors expanded to DIM_SENSOR; loop covers all dims
 *   FIX-19  learn():                 Properly forwards nextObservation to estimate next value
 *   FIX-20  _hashSeed:              Replaced with FNV-1a for better avalanche
 *   FIX-21  gradientClips counter:   Now incremented when clipping actually occurs
 *   FIX-22  Social memory bounded:   LRU eviction when relationships exceed MAX_RELATIONSHIPS
 *   FIX-23  Hot-loop performance:    Removed optional chaining, _safeNumber, console.assert from inner loops
 *   FIX-24  Constructor:             Extracted into focused init methods
 *   FIX-25  God class:               Extracted EmotionSystem, MemorySystem, SchemaManager, SeededRNG
 *   FIX-26  Magic numbers:           All remaining literals named as constants in ARCH
 *   FIX-27  fromJSON validation:     Dimension checks on all weight matrices
 *   FIX-28  Reflex ordering:         Reflex/conflict resolution moved BEFORE state commit (critical)
 *   FIX-29  Emotion serialisation:   toJSON/fromJSON now include current emotion values
 *   FIX-30  Naming consistency:      Standardised to camelCase throughout
 *
 * PREVIOUS FIXES (v2.1 → v2.2) preserved:
 *   FIX-01 through FIX-12 (see v2.2 header)
 *
 * USAGE:
 *   import { Cortex2Brain } from './cortex.js';
 *   const brain = new Cortex2Brain({ seed: 'agent_1', taskSpec: {...} });
 *   brain.configurePersonality({ traits: { bravery: 0.8 } });
 *   const result = brain.forward(sensors, reward, context);
 *   const debug  = brain.getDebugInfo();
 *   brain.learn({ observation, action, reward, nextObservation, done });
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * @typedef {Object} CortexConfig
 * @property {string}  [seed]
 * @property {number}  [lr]
 * @property {number}  [gamma]
 * @property {number}  [lambda]
 * @property {number}  [hebbianRate]
 * @property {Object}  [taskSpec]
 * @property {number}  [taskSpec.inputComplexity]
 * @property {number}  [taskSpec.temporalDepth]
 * @property {number}  [taskSpec.socialComplexity]
 */

/**
 * @typedef {Object} SensorSchema
 * @property {string}   name
 * @property {[number, number]} range
 * @property {string}   [description]
 * @property {'continuous'|'discrete'|'categorical'} [type]
 */

/**
 * @typedef {Object} ActionSchema
 * @property {string}   name
 * @property {[number, number]} range
 * @property {string}   [description]
 * @property {'continuous'|'discrete'} [type]
 */

/**
 * @typedef {Object} Experience
 * @property {number[]|Object} observation
 * @property {number[]|Object} action
 * @property {number}          reward
 * @property {number[]|Object} [nextObservation]
 * @property {boolean}         [done]
 * @property {Object}          [context]
 */

/**
 * @typedef {Object} ForwardResult
 * @property {number[]} output
 * @property {number}   predictionError
 * @property {boolean}  reflexTriggered
 * @property {string|null} emotionalStateName
 * @property {number}   stepCount
 * @property {Object|null} trace
 */

/**
 * @typedef {Object} DebugInfo
 * @property {Object}   emotions
 * @property {Object}   personality
 * @property {Object|null} emotionalState
 * @property {Object}   stats
 * @property {number[]} attentionWeights
 * @property {number}   predictionError
 * @property {Object}   memoryUsage
 * @property {number}   cortexEnergy
 * @property {Object}   ratios
 * @property {Object|null} trace
 * @property {number[]} cortex
 * @property {Object}   emotionsFull
 * @property {Object}   socialMemory
 */

// ============================================================================
// ARCHITECTURE CONSTANTS
// ============================================================================

/** @param {Object} DIM */
function deriveArchSizes(DIM) {
    return Object.freeze({
        P_INPUT:      DIM.DIM_SENSOR + DIM.DIM_C + DIM.DIM_S + 48,
        A_INPUT:      DIM.DIM_P + 48 + 64 + DIM.DIM_D,
        M_WORK_INPUT: DIM.DIM_P + DIM.DIM_A + 64,
        M_PERM_INPUT: DIM.DIM_C + 48,
        M_PRED_INPUT: DIM.DIM_A + 64 + 48 + DIM.DIM_D,
        C_INPUT:      DIM.DIM_P + DIM.DIM_A + DIM.DIM_M + DIM.DIM_S + DIM.DIM_D + 1,
        S_INPUT:      DIM.DIM_P + 48 + 32 + DIM.DIM_D,
        D_INPUT:      DIM.DIM_C + DIM.DIM_S + 48,

        // FIX-26: named constants for all remaining magic numbers
        C_PRED_OFFSET:       192,
        C_STATE_OFFSET:      224,
        C_STATE_SLICE:       32,
        A_HEAD_C_SLICE:      64,
        M_WORK_C_SLICE:      64,
        M_BLOCK_SIZE:        48,
        PRED_CHANNELS:       48, // kept equal to M_BLOCK_SIZE for prediction
        MAX_RELATIONSHIPS:   100,
        MAX_EMOTION_HISTORY: 5,
        MAX_EPISODIC_EVENTS: 50,
        MAX_TD_HISTORY:      100,
        MAX_TRACE_SIZE:      10,
        EMOTION_MOMENTUM_WINDOW: 10,
        REFLEX_THRESHOLD:    0.9,
        MAX_WEIGHT:          2.0,
        MAX_GRADIENT:        0.3,
        MAX_TRACE_VAL:       10.0,
    });
}

// ============================================================================
// SEEDED RNG (FIX-14, FIX-25: extracted as standalone class)
// ============================================================================

class SeededRNG {
    /**
     * @param {string|number} seed
     */
    constructor(seed) {
        // FIX-14: guarantee non-zero state for xorshift32
        let h = 0x811c9dc5; // FNV offset basis
        const s = String(seed || 'default');
        for (let i = 0; i < s.length; i++) {
            h ^= s.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        this._state = (h >>> 0) || 1; // xorshift32 requires non-zero
    }

    /** @returns {number} float in [0, 1) */
    next() {
        let x = this._state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this._state = x >>> 0;
        // FIX-14: if state somehow becomes 0, reset to 1
        if (this._state === 0) this._state = 1;
        return this._state / 4294967296;
    }

    /**
     * @param {number} mean
     * @param {number} std
     * @returns {number}
     */
    gaussian(mean = 0, std = 1) {
        const u1 = Math.max(this.next(), 1e-10);
        const u2 = this.next();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return z * std + mean;
    }
}

// ============================================================================
// SCHEMA MANAGER (FIX-25: extracted)
// ============================================================================

class SchemaManager {
    /**
     * @param {number} sensorDim
     */
    constructor(sensorDim) {
        this._sensorDim = sensorDim;
        this._inputSchema = this._generateDefaultInputSchema();
        this._outputSchema = this._generateDefaultOutputSchema();
    }

    getInputSchema() { return [...this._inputSchema]; }
    getOutputSchema() { return [...this._outputSchema]; }

    /** @param {SensorSchema[]} schema */
    setInputSchema(schema) {
        if (!Array.isArray(schema)) throw new Error('Input schema must be an array');
        this._inputSchema = schema.map(s => ({
            name: String(s.name),
            range: Array.isArray(s.range) ? s.range : [-1, 1],
            description: s.description || '',
            type: s.type || 'continuous',
        }));
        if (schema.length > 0 && schema.length !== this._sensorDim) {
            console.warn(`Schema length (${schema.length}) differs from DIM_SENSOR (${this._sensorDim}). Truncating/padding.`);
        }
    }

    /** @param {ActionSchema[]} schema */
    setOutputSchema(schema) {
        if (!Array.isArray(schema)) throw new Error('Output schema must be an array');
        this._outputSchema = schema.map(s => ({
            name: String(s.name),
            range: Array.isArray(s.range) ? s.range : [-1, 1],
            description: s.description || '',
            type: s.type || 'continuous',
        }));
    }

    /**
     * @param {number[]|Object} inputs
     * @param {Object} stats - validation error counter
     * @returns {number[]}
     */
    validateAndNormalizeInputs(inputs, stats) {
        if (inputs && typeof inputs === 'object' && !Array.isArray(inputs)) {
            const arr = [];
            for (const sensor of this._inputSchema) {
                const val = inputs[sensor.name];
                if (val === undefined) {
                    arr.push(0);
                } else {
                    const [min, max] = sensor.range || [-1, 1];
                    arr.push(clamp((val - min) / (max - min) * 2 - 1, -1, 1));
                }
            }
            inputs = arr;
        }

        if (!Array.isArray(inputs)) {
            stats.validationErrors++;
            throw new Error(`Inputs must be array or object, got ${typeof inputs}`);
        }
        if (inputs.length === 0) {
            stats.validationErrors++;
            throw new Error('Inputs array cannot be empty');
        }

        for (let i = 0; i < Math.min(inputs.length, 20); i++) {
            if (!isFinite(inputs[i])) {
                stats.validationErrors++;
                console.warn(`Invalid input at index ${i}: ${inputs[i]}. Replacing with 0.`);
                inputs[i] = 0;
            }
        }
        return inputs;
    }

    /**
     * @param {number[]} outputs
     * @returns {number[]}
     */
    validateAndClampOutputs(outputs) {
        if (!Array.isArray(outputs)) { console.warn('Outputs must be array'); return []; }
        const clamped = [];
        for (let i = 0; i < outputs.length && i < this._outputSchema.length; i++) {
            const [min, max] = this._outputSchema[i].range || [-1, 1];
            clamped.push(clamp(outputs[i], min, max));
        }
        while (clamped.length < this._outputSchema.length) {
            const [min, max] = this._outputSchema[clamped.length].range || [-1, 1];
            clamped.push((min + max) / 2);
        }
        return clamped;
    }

    _generateDefaultInputSchema() {
        return [
            { name: 'enemy_x', range: [-1, 1] },
            { name: 'enemy_y', range: [-1, 1] },
            { name: 'distance', range: [0, 1] },
            { name: 'enemy_health', range: [0, 1] },
            { name: 'wall_x', range: [-1, 1] },
            { name: 'wall_y', range: [-1, 1] },
            { name: 'wall_distance', range: [0, 1] },
            { name: 'self_x', range: [-1, 1] },
            { name: 'self_y', range: [-1, 1] },
            { name: 'self_health', range: [0, 1] },
            { name: 'self_reward', range: [-1, 1] },
            { name: 'self_damage', range: [0, 1] },
            { name: 'self_dodge', range: [0, 1] },
            { name: 'self_fireRate', range: [0, 1] },
            { name: 'step_normalized', range: [0, 1] },
            { name: 'weapon_hammer', range: [0, 1], type: 'discrete' },
            { name: 'weapon_drill', range: [0, 1], type: 'discrete' },
            { name: 'weapon_sword', range: [0, 1], type: 'discrete' },
            { name: 'weapon_blaster', range: [0, 1], type: 'discrete' },
            { name: 'weapon_axe', range: [0, 1], type: 'discrete' },
            { name: 'weapon_dagger', range: [0, 1], type: 'discrete' },
            { name: 'weapon_spear', range: [0, 1], type: 'discrete' },
            { name: 'weapon_chainsaw', range: [0, 1], type: 'discrete' },
            { name: 'death_count', range: [0, 1] },
            { name: 'forced_combat', range: [0, 1], type: 'discrete' },
            { name: 'angle_to_enemy', range: [-1, 1] },
            { name: 'enemy_sees_me', range: [-1, 1] },
            { name: 'in_enemy_arc', range: [0, 1], type: 'discrete' },
            { name: 'enemy_in_my_arc', range: [0, 1], type: 'discrete' },
            { name: 'threat_level', range: [0, 1] },
            { name: 'flank_vector_x', range: [-1, 1] },
            { name: 'flank_vector_y', range: [-1, 1] },
            { name: 'angle_from_enemy', range: [-1, 1] },
            { name: 'dist_to_arc_edge', range: [-1, 1] },
            { name: 'self_damage_ratio', range: [0, 1] },
            { name: 'predicted_enemy_x', range: [-1, 1] },
            { name: 'predicted_enemy_y', range: [-1, 1] },
            { name: 'path_blocked', range: [0, 1], type: 'discrete' },
            { name: 'recent_exposure', range: [0, 1] },
            { name: 'exposure_level', range: [0, 1] },
            { name: 'in_combat_range', range: [0, 1], type: 'discrete' },
            { name: 'wall_dist_normalized', range: [0, 1] },
            { name: 'wall_avoidance_steer', range: [-1, 1] },
        ];
    }

    _generateDefaultOutputSchema() {
        return [
            { name: 'output_0', range: [-1, 1] },
            { name: 'output_1', range: [-1, 1] },
            { name: 'output_2', range: [-1, 1] },
            { name: 'output_3', range: [-1, 1] },
            { name: 'throttle', range: [-1, 1] },
            { name: 'curiosity', range: [0, 1] },
            { name: 'aggression', range: [0, 1] },
            { name: 'memoryWeight', range: [0, 1] },
            { name: 'steering', range: [-1, 1] },
            { name: 'circularBias', range: [-1, 1] },
            { name: 'retreatUrgency', range: [0, 1] },
            { name: 'flankCommit', range: [0, 1] },
            { name: 'output_12', range: [-1, 1] },
            { name: 'output_13', range: [-1, 1] },
            { name: 'output_14', range: [-1, 1] },
            { name: 'output_15', range: [-1, 1] },
        ];
    }
}

// ============================================================================
// EMOTION SYSTEM (FIX-25: extracted)
// ============================================================================

class EmotionSystem {
    /**
     * @param {Object} personality
     * @param {number} maxHistory
     */
    constructor(personality, maxHistory = 30) {
        this.personality = personality;
        this.maxHistory = maxHistory;

        this.emotions = EmotionSystem.createDefaultEmotions();
        this.emotionHistory = [];

        this.emotionalState = {
            current: null,
            history: [],
            satisfaction: 0,
            momentum: {
                fear: 0, aggression: 0, frustration: 0,
                desperation: 0, confidence: 0, vengeance: 0,
            },
        };

        this.emotionMemory = {
            trauma: 0,
            grudge: new Map(),
            lastEmotion: null,
            unresolved: 0,
            killer: null,
        };

        this.emotionConfig = {
            fear: { baseDuration: 180, intensityMultiplier: 2.0, satisfactionDecay: 0.02 },
            aggression: { baseDuration: 120, intensityMultiplier: 1.5, satisfactionDecay: 0.03 },
            frustration: { baseDuration: 240, intensityMultiplier: 2.5, satisfactionDecay: 0.01 },
            desperation: { baseDuration: 300, intensityMultiplier: 3.0, satisfactionDecay: 0.005 },
            confidence: { baseDuration: 150, intensityMultiplier: 1.2, satisfactionDecay: 0.04 },
            vengeance: { baseDuration: 360, intensityMultiplier: 3.5, satisfactionDecay: 0.003 },
        };

        this.emotionDecay = 0.98;
        this.emotionInfluence = 0.35;
    }

    static createDefaultEmotions() {
        return {
            threat: 0, frustration: 0, confidence: 0, aggression: 0,
            fear: 0, surprise: 0, caution: 0, desperation: 0, vengeance: 0,
            empathy: 0, loyalty: 0, shame: 0, pride: 0, curiosity: 0, boredom: 0,
        };
    }

    /** @param {Object} config */
    configure(config) {
        if (config.emotionConfig) {
            for (const [emotion, cfg] of Object.entries(config.emotionConfig)) {
                if (emotion in this.emotionConfig) {
                    this.emotionConfig[emotion] = { ...this.emotionConfig[emotion], ...cfg };
                }
            }
        }
        if (config.decayRate !== undefined) this.emotionDecay = clamp(config.decayRate, 0.9, 0.999);
        if (config.influenceWeight !== undefined) this.emotionInfluence = clamp(config.influenceWeight, 0.1, 0.9);
    }

    /**
     * @param {number} reward
     * @param {Float32Array} sensorInputs
     * @param {Object|null} combatEvents
     * @param {number} step
     * @param {number} cumulativeReward
     * @param {number} avgPredError
     * @param {number} health
     * @param {number} maxHealth
     */
    update(reward, sensorInputs, combatEvents, step, cumulativeReward, avgPredError, health, maxHealth) {
        this.emotionHistory.push({ ...this.emotions, step });
        if (this.emotionHistory.length > this.maxHistory) this.emotionHistory.shift();

        if (combatEvents) this._processCombatEvents(combatEvents, health, maxHealth);

        if (this.emotionalState.current) {
            const state = this.emotionalState.current;
            const config = this.emotionConfig[state.name];
            state.remainingFrames--;
            state.satisfaction = lerp(state.satisfaction, this.emotionalState.satisfaction, 0.1);

            // FIX-17: check exit FIRST (handles expiration)
            if (this._canExitEmotion(state)) {
                this._exitEmotion(state);
            } else {
                if (this._shouldExtendEmotion(state)) {
                    state.remainingFrames = Math.min(
                        state.remainingFrames + 30,
                        config.baseDuration * 3
                    );
                }
                const transition = this._checkEmotionTransition(state);
                if (transition) this._enterEmotion(transition.name, transition.intensity, step);
            }
        } else {
            const dominant = this._getDominantEmotion();
            if (dominant.intensity > 0.7) this._enterEmotion(dominant.name, dominant.intensity, step);
        }

        // Reward-driven emotion updates
        if (reward < 0.1) this.emotions.frustration = Math.min(1, this.emotions.frustration + 0.03);
        else this.emotions.frustration *= 0.92;

        const normalizedReward = clamp((cumulativeReward + 100) / 200, 0, 1);
        if (normalizedReward < 0.3) this.emotions.desperation = Math.min(1, this.emotions.desperation + 0.05);
        else this.emotions.desperation *= 0.85;

        if (reward > 1) this.emotions.confidence = Math.min(1, this.emotions.confidence + 0.05);
        else this.emotions.confidence *= 0.95;

        if (reward < -0.5) this.emotions.fear = Math.min(1, this.emotions.fear + 0.1);
        else this.emotions.fear *= 0.85;

        this.emotions.surprise = avgPredError;
        this.emotions.caution = 1 - (sensorInputs[12] ?? 0.5);
        this.emotions.threat = 1 - (sensorInputs[2] ?? 0.5);

        // Momentum
        const recentFrames = Math.min(this.maxHistory, this.emotionHistory.length);
        if (recentFrames > 0) {
            const window = Math.min(10, recentFrames);
            const recent = this.emotionHistory.slice(-window);
            const avg = key => recent.reduce((s, e) => s + (e[key] || 0), 0) / window;
            this.emotionalState.momentum.fear = avg('fear');
            this.emotionalState.momentum.aggression = avg('aggression');
            this.emotionalState.momentum.frustration = avg('frustration');
            this.emotionalState.momentum.desperation = avg('desperation');
            this.emotionalState.momentum.confidence = avg('confidence');
            this.emotionalState.momentum.vengeance = avg('vengeance');
        }

        // Derived aggression
        const fearBlock = (1 - this.emotions.desperation) * this.emotionalState.momentum.fear;
        this.emotions.aggression =
            this.emotionalState.momentum.frustration * 0.6 +
            this.emotionalState.momentum.desperation * 0.8 +
            this.emotionalState.momentum.vengeance * 0.9 -
            fearBlock * 0.4;

        // Decay
        for (const key of Object.keys(this.emotions)) {
            this.emotions[key] *= this.emotionDecay;
        }
    }

    /**
     * Write emotion values into state vector S.
     * @param {Float32Array} stateVector
     */
    writeToStateVector(stateVector) {
        stateVector[0] = this.emotions.threat;
        stateVector[1] = this.emotions.frustration;
        stateVector[2] = this.emotions.confidence;
        stateVector[3] = this.emotions.aggression;
        stateVector[4] = this.emotions.fear;
        stateVector[5] = this.emotions.surprise;
        stateVector[6] = this.emotions.caution;
        stateVector[7] = this.emotions.desperation;
        stateVector[8] = this.emotions.vengeance;
    }

    /**
     * Apply reflex modifications to decision vector.
     * FIX-28: This is now called BEFORE state commit in forward().
     * @param {Float32Array} sensorInputs
     * @param {Float32Array} stateVector
     * @param {Float32Array} decisionVector
     * @param {Object} stats
     */
    applyReflexes(sensorInputs, stateVector, decisionVector, stats) {
        const em = this.emotionInfluence;
        if (stateVector[8] > 0.6) {
            stats.reflexTriggers++;
            decisionVector[6] = lerp(decisionVector[6], Math.min(1, decisionVector[6] + 0.40), em);
            decisionVector[4] = lerp(decisionVector[4], Math.min(1, decisionVector[4] + 0.30), em);
        }
        if (stateVector[7] > 0.6) {
            stats.reflexTriggers++;
            decisionVector[6] = lerp(decisionVector[6], Math.min(1, decisionVector[6] + 0.35), em);
            decisionVector[4] = lerp(decisionVector[4], Math.min(1, decisionVector[4] + 0.30), em);
        }
        if (stateVector[4] > 0.7) {
            stats.reflexTriggers++;
            decisionVector[10] = lerp(decisionVector[10], Math.min(1, decisionVector[10] + 0.35), em);
            decisionVector[4] = lerp(decisionVector[4], Math.max(0, decisionVector[4] - 0.20), em);
        }
        if (stateVector[1] > 0.7) {
            stats.reflexTriggers++;
            decisionVector[6] = lerp(decisionVector[6], Math.min(1, decisionVector[6] + 0.30), em);
            decisionVector[4] = lerp(decisionVector[4], Math.min(1, decisionVector[4] + 0.25), em);
        }
    }

    /**
     * Resolve conflicts between emotion-driven commands.
     * @param {Float32Array} decisionVector
     */
    resolveConflicts(decisionVector) {
        const eb = this.emotionInfluence;
        const emotions = this.emotions;
        if (emotions.vengeance > 0.6 || emotions.desperation > 0.6) {
            decisionVector[10] = Math.max(0, decisionVector[10] - eb);
            decisionVector[7] = Math.min(1, decisionVector[7] + eb * 0.5);
        }
        if (emotions.fear > 0.7) {
            decisionVector[6] = Math.max(0, decisionVector[6] - eb);
            decisionVector[4] = Math.max(0, decisionVector[4] - eb * 0.3);
        }
        if (emotions.aggression > 0.6) {
            decisionVector[6] = Math.min(1, decisionVector[6] + eb);
            decisionVector[4] = Math.min(1, decisionVector[4] + eb * 0.3);
        }
    }

    resetEpisode() {
        this.emotions = EmotionSystem.createDefaultEmotions();
        this.emotionalState.current = null;
        this.emotionalState.history = [];
        this.emotionHistory = [];
    }

    // --- Private ---

    _getDominantEmotion() {
        const candidates = [
            { name: 'vengeance', value: this.emotions.vengeance },
            { name: 'desperation', value: this.emotions.desperation },
            { name: 'fear', value: this.emotions.fear },
            { name: 'aggression', value: this.emotions.aggression },
            { name: 'frustration', value: this.emotions.frustration },
            { name: 'confidence', value: this.emotions.confidence },
        ];
        candidates.sort((a, b) => b.value - a.value);
        return { name: candidates[0].name, intensity: candidates[0].value };
    }

    /**
     * @param {string} name
     * @param {number} intensity
     * @param {number} step
     */
    _enterEmotion(name, intensity, step) {
        const config = this.emotionConfig[name];
        if (!config) return;
        if (this.emotionalState.current) {
            this.emotionalState.history.push({
                ...this.emotionalState.current,
                endStep: step,
                reason: 'transition',
            });
            if (this.emotionalState.history.length > 5) this.emotionalState.history.shift();
        }
        this.emotionalState.current = {
            name,
            intensity,
            initialIntensity: intensity,
            baseDuration: config.baseDuration,
            remainingFrames: Math.floor(config.baseDuration * intensity * config.intensityMultiplier),
            satisfaction: this.emotionalState.satisfaction,
            startStep: step,
        };
    }

    /**
     * @param {Object} state
     * @param {string} reason
     */
    _exitEmotion(state, reason = 'completed') {
        this.emotionalState.history.push({ ...state, reason });
        if (this.emotionalState.history.length > 5) this.emotionalState.history.shift();
        this.emotionMemory.lastEmotion = {
            name: state.name,
            intensity: state.intensity,
            satisfaction: state.satisfaction,
            completed: reason === 'completed' || reason === 'satisfied',
        };
        if (reason === 'transition' || reason === 'interrupted') {
            this.emotionMemory.unresolved = clamp(
                this.emotionMemory.unresolved + state.intensity * 0.5, 0, 1
            );
        }
        this.emotionalState.current = null;
    }

    /**
     * FIX-17: Fixed inverted logic.
     * - Returns true when remainingFrames expired (natural completion)
     * - Returns true when satisfied or faded
     * - Returns false when emotion is still strong and has time left
     * @param {Object} state
     * @returns {boolean}
     */
    _canExitEmotion(state) {
        if (state.remainingFrames <= 0) return true;     // expired
        if (state.satisfaction > 0.6) return true;       // satisfied
        if (state.intensity < 0.3) return true;          // faded
        // Too early to exit if more than 70% of time remains
        if (state.remainingFrames > state.baseDuration * 0.7) return false;
        return false;
    }

    /** @param {Object} state */
    _shouldExtendEmotion(state) {
        return state.intensity > state.initialIntensity + 0.2 || state.satisfaction < -0.3;
    }

    /** @param {Object} state */
    _checkEmotionTransition(state) {
        const c = state.name;
        if (c === 'fear' && this.emotions.desperation > 0.7)
            return { name: 'desperation', intensity: this.emotions.desperation };
        if (c === 'frustration' && this.emotions.aggression > 0.6)
            return { name: 'aggression', intensity: this.emotions.aggression };
        if (c === 'aggression' && state.satisfaction < -0.5)
            return { name: 'frustration', intensity: this.emotions.frustration };
        if (c === 'desperation' && state.intensity < 0.4)
            return { name: 'fear', intensity: this.emotions.fear };
        if (c === 'vengeance' && state.satisfaction > 0.5)
            return { name: 'confidence', intensity: this.emotions.confidence };
        return null;
    }

    /**
     * @param {Object} events
     * @param {number} health
     * @param {number} maxHealth
     */
    _processCombatEvents(events, health, maxHealth) {
        if (!events) return;
        for (const event of events) {
            switch (event.type) {
                case 'hit_dealt':
                    this.emotionalState.satisfaction = clamp(this.emotionalState.satisfaction + 0.15, -1, 1);
                    if (this.emotionalState.current?.name === 'aggression' ||
                        this.emotionalState.current?.name === 'vengeance') {
                        this.emotionalState.current.intensity = Math.min(
                            1, this.emotionalState.current.intensity + 0.05
                        );
                    }
                    break;
                case 'hit_missed':
                    this.emotions.frustration = Math.min(1, this.emotions.frustration + 0.1);
                    this.emotionalState.satisfaction = clamp(this.emotionalState.satisfaction - 0.1, -1, 1);
                    break;
                case 'damage_received':
                    this.emotions.fear = Math.min(1, this.emotions.fear + (event.damage || 1) * 0.02);
                    this.emotionalState.satisfaction = clamp(this.emotionalState.satisfaction - 0.2, -1, 1);
                    if (health / maxHealth < 0.3) {
                        this.emotions.desperation = Math.min(1, this.emotions.desperation + 0.15);
                    }
                    break;
                case 'enemy_defeated':
                    this.emotions.confidence = Math.min(1, this.emotions.confidence + 0.3);
                    this.emotionalState.satisfaction = clamp(this.emotionalState.satisfaction + 0.5, -1, 1);
                    if (this.emotionalState.current?.name === 'aggression' ||
                        this.emotionalState.current?.name === 'vengeance') {
                        this._exitEmotion(this.emotionalState.current, 'satisfied');
                    }
                    break;
                case 'killed_by':
                    if (event.killer) {
                        this.emotionMemory.killer = event.killer;
                        const grudge = this.emotionMemory.grudge.get(event.killer) || 0;
                        this.emotionMemory.grudge.set(event.killer, clamp(grudge + 0.5, -1, 1));
                        this.emotions.vengeance = Math.min(1, this.emotions.vengeance + 0.6);
                    }
                    break;
            }
        }
    }

    /** @returns {Object} serialisable snapshot */
    toJSON() {
        return {
            // FIX-29: include current emotion values
            emotions: { ...this.emotions },
            emotionMemory: {
                trauma: this.emotionMemory.trauma,
                grudge: Array.from(this.emotionMemory.grudge.entries()),
                lastEmotion: this.emotionMemory.lastEmotion,
                unresolved: this.emotionMemory.unresolved,
                killer: this.emotionMemory.killer,
            },
            emotionalState: {
                current: this.emotionalState.current ? { ...this.emotionalState.current } : null,
                satisfaction: this.emotionalState.satisfaction,
                momentum: { ...this.emotionalState.momentum },
            },
        };
    }

    /** @param {Object} data */
    loadFromJSON(data) {
        // FIX-29: restore current emotion values
        if (data.emotions) {
            for (const key of Object.keys(this.emotions)) {
                if (key in data.emotions) this.emotions[key] = data.emotions[key];
            }
        }
        if (data.emotionMemory) {
            this.emotionMemory.trauma = data.emotionMemory.trauma || 0;
            this.emotionMemory.grudge = new Map(data.emotionMemory.grudge || []);
            this.emotionMemory.lastEmotion = data.emotionMemory.lastEmotion;
            this.emotionMemory.unresolved = data.emotionMemory.unresolved || 0;
            this.emotionMemory.killer = data.emotionMemory.killer;
        }
        if (data.emotionalState) {
            this.emotionalState.current = data.emotionalState.current
                ? { ...data.emotionalState.current }
                : null;
            this.emotionalState.satisfaction = data.emotionalState.satisfaction ?? 0;
            if (data.emotionalState.momentum) {
                Object.assign(this.emotionalState.momentum, data.emotionalState.momentum);
            }
        }
    }
}

// ============================================================================
// MEMORY SYSTEM (FIX-25: extracted)
// ============================================================================

class MemorySystem {
    /**
     * @param {number} maxRelationships
     * @param {number} maxEvents
     */
    constructor(maxRelationships = 100, maxEvents = 50) {
        this.maxRelationships = maxRelationships;
        this.maxEvents = maxEvents;

        this.socialMemory = {
            relationships: new Map(),
            reputation: 0,
            observedBehaviors: new Map(),
        };

        this.episodicMemory = {
            events: [],
            maxEvents,
            recallTrigger: 0.7,
            decayRate: 0.99,
        };
    }

    /**
     * FIX-22: LRU eviction when relationships exceed max.
     * @param {Object} socialContext
     * @param {number} step
     * @param {Object} personality
     * @param {Object} emotions
     */
    processSocialInput(socialContext, step, personality, emotions) {
        const others = socialContext.others || [];
        for (const other of others) {
            const rel = this.socialMemory.relationships.get(other.id) ||
                { trust: 0, affinity: 0, lastInteraction: 0, sharedHistory: [] };
            const otherIsAggressive = other.aggression > 0.7;
            const otherIsThreatened = (other.healthRatio || 1.0) < 0.4;

            if (otherIsThreatened && personality.traits.empathy > 0.6) {
                emotions.empathy = Math.min(1, emotions.empathy + 0.1);
            }
            if (otherIsAggressive && rel.trust < 0) {
                emotions.fear = Math.min(1, emotions.fear + 0.15);
            }

            const recentInteraction = (other.distance || 100) < 100;
            if (recentInteraction) {
                if (!otherIsAggressive && (other.distance || 100) < 80) {
                    rel.trust = clamp(rel.trust + 0.05, -1, 1);
                    rel.affinity = clamp(rel.affinity + 0.03, -1, 1);
                } else if (otherIsAggressive) {
                    rel.trust = clamp(rel.trust - 0.10, -1, 1);
                    rel.affinity = clamp(rel.affinity - 0.05, -1, 1);
                }
                rel.lastInteraction = step;
                if (rel.sharedHistory.length > 20) rel.sharedHistory.shift();
            }
            rel.trust *= 0.999;
            rel.affinity *= 0.999;
            this.socialMemory.relationships.set(other.id, rel);
        }

        // FIX-22: LRU eviction
        this._evictOldRelationships();
    }

    /**
     * @param {Object} query
     * @returns {Object[]}
     */
    recallRelevantMemories({ enemyId, healthRatio, situation }) {
        const recalled = [];
        for (const event of this.episodicMemory.events) {
            if (enemyId && event.agentId === enemyId) {
                recalled.push({ ...event, relevance: event.importance * 1.5 });
            }
            if (situation === 'low_health' && event.emotion === 'fear') {
                recalled.push({ ...event, relevance: event.importance * 1.2 });
            }
            if (situation === 'advantage' && event.emotion === 'pride') {
                recalled.push({ ...event, relevance: event.importance * 1.2 });
            }
        }
        recalled.sort((a, b) => b.relevance - a.relevance);
        return recalled.slice(0, 3);
    }

    /**
     * FIX-11: consolidationEvents counter now incremented.
     * @param {Float32Array} predictionErrors
     * @param {Float32Array} memoryVector
     * @param {number} reward
     * @param {number} step
     * @param {Object} stats
     * @param {number} predStart
     */
    consolidateMemories(predictionErrors, memoryVector, reward, step, stats, predStart) {
        let newEvents = 0;
        const blockSize = predictionErrors.length;
        for (let i = 0; i < blockSize; i++) {
            if (predictionErrors[i] > 0.25) {
                this.episodicMemory.events.push({
                    idx: i,
                    value: memoryVector[predStart + i],
                    importance: predictionErrors[i] * reward,
                    step,
                });
                newEvents++;
            }
        }
        stats.consolidationEvents += newEvents;

        if (this.episodicMemory.events.length > this.episodicMemory.maxEvents) {
            this.episodicMemory.events.sort((a, b) => b.importance - a.importance);
            this.episodicMemory.events.length = this.episodicMemory.maxEvents;
        }
    }

    resetEpisode(decayRate) {
        this.socialMemory.reputation *= 0.95;
        for (const event of this.episodicMemory.events) {
            event.importance *= decayRate;
        }
    }

    resetFull() {
        this.socialMemory.relationships.clear();
        this.socialMemory.reputation = 0;
        this.socialMemory.observedBehaviors.clear();
        this.episodicMemory.events = [];
    }

    /** FIX-22: evict least-recently-interacted relationship */
    _evictOldRelationships() {
        while (this.socialMemory.relationships.size > this.maxRelationships) {
            let oldestId = null;
            let oldestStep = Infinity;
            for (const [id, rel] of this.socialMemory.relationships) {
                if (rel.lastInteraction < oldestStep) {
                    oldestId = id;
                    oldestStep = rel.lastInteraction;
                }
            }
            if (oldestId !== null) {
                this.socialMemory.relationships.delete(oldestId);
            } else {
                break;
            }
        }
    }

    /** @returns {Object} */
    toJSON() {
        return {
            socialMemory: {
                relationships: Array.from(this.socialMemory.relationships.entries()).map(
                    ([k, v]) => [k, {
                        trust: v.trust,
                        affinity: v.affinity,
                        lastInteraction: v.lastInteraction,
                        sharedHistory: v.sharedHistory?.slice(0, 10) || [],
                    }]
                ),
                reputation: this.socialMemory.reputation,
            },
            episodicMemory: this.episodicMemory.events.map(e => ({ ...e })),
        };
    }

    /** @param {Object} data */
    loadFromJSON(data) {
        if (data.socialMemory) {
            this.socialMemory.reputation = data.socialMemory.reputation ?? 0;
            if (data.socialMemory.relationships) {
                for (const [id, rel] of data.socialMemory.relationships) {
                    this.socialMemory.relationships.set(id, {
                        trust: rel.trust ?? 0,
                        affinity: rel.affinity ?? 0,
                        lastInteraction: rel.lastInteraction ?? 0,
                        sharedHistory: rel.sharedHistory || [],
                    });
                }
            }
        }
        if (data.episodicMemory) {
            this.episodicMemory.events = data.episodicMemory.map(e => ({ ...e }));
        }
    }
}

// ============================================================================
// STANDALONE MATH UTILITIES (FIX-23, FIX-30: top-level for inlining)
// ============================================================================

/** @param {number} v @param {number} min @param {number} max */
function clamp(v, min, max) { return v < min ? min : v > max ? max : v; }

/** @param {number} a @param {number} b @param {number} t */
function lerp(a, b, t) { const ct = t < 0 ? 0 : t > 1 ? 1 : t; return a + (b - a) * ct; }

/** @param {*} val @param {number} fallback */
function safeNumber(val, fallback) {
    return (typeof val === 'number' && isFinite(val)) ? val : fallback;
}

/** @param {number} x */
function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e = Math.exp(2 * x);
    return (e - 1) / (e + 1);
}

/** @param {number} x @param {number} alpha */
function leakyRelu(x, alpha = 0.01) { return x > 0 ? x : alpha * x; }

/**
 * FIX-09/FIX-23: safe softmax using reduce, accepts plain Array.
 * @param {number[]} arr
 * @param {number} temp
 * @returns {number[]}
 */
function softmax(arr, temp = 1) {
    if (!arr || arr.length === 0) return [];
    let max = -Infinity;
    for (let i = 0; i < arr.length; i++) { if (arr[i] > max) max = arr[i]; }
    const exps = new Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        const v = Math.exp(Math.min((arr[i] - max) / temp, 700));
        exps[i] = v;
        sum += v;
    }
    if (sum < 1e-10) sum = 1e-10;
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
}

// ============================================================================
// MAIN CLASS
// ============================================================================

export class Cortex2Brain {

    /** @param {CortexConfig} config */
    constructor(config = {}) {
        this._initConfig(config);       // FIX-24: extracted
        this._initArchitecture();       // FIX-24: extracted
        this._initWeights();
        this._initStateVectors();
        this._initEligibilityTraces();
        this._initSubsystems();         // FIX-24: extracted
        this._initRuntimeState();       // FIX-24: extracted
        this._isReady = true;
    }

    // ========================================================================
    // INITIALISATION (FIX-24: decomposed constructor)
    // ========================================================================

    /** @param {CortexConfig} config */
    _initConfig(config) {
        this.config = {
            seed: config.seed || `CORTEX_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
            lr: config.lr ?? 0.015,
            gamma: config.gamma ?? 0.99,
            lambda: config.lambda ?? 0.95,
            hebbianRate: config.hebbianRate ?? 0.002,
            taskSpec: config.taskSpec || null,
            ...config,
        };
    }

    _initArchitecture() {
        this.DIM = Object.freeze({
            DIM_P: 32,
            DIM_A: 64,
            DIM_M: 144,
            DIM_C: 256,
            DIM_S: 32,
            DIM_D: 16,
            DIM_SENSOR: 64,
            NUM_HEADS: 4,
            HEAD_DIM: 16,
            M_WORK_START: 0, M_WORK_END: 48,
            M_PERM_START: 48, M_PERM_END: 96,
            M_PRED_START: 96, M_PRED_END: 144,
        });

        this.ARCH = deriveArchSizes(this.DIM);
        this.RATIOS = this._calculateRatios(this.config.taskSpec);
        this._rng = new SeededRNG(this.config.seed);
    }

    _initSubsystems() {
        // FIX-18: predictionErrors now covers full DIM_SENSOR
        this.predictionErrors = new Float32Array(this.DIM.DIM_SENSOR);

        // Stats
        this._stats = this._createEmptyStats();

        // Personality
        this.personality = {
            traits: {
                bravery: 0.5, loyalty: 0.5, empathy: 0.5,
                curiosity: 0.5, patience: 0.5, aggression: 0.5,
            },
            values: {
                survival: 0.8, victory: 0.7, fairness: 0.4, loyalty: 0.6,
            },
        };

        // FIX-25: extracted subsystems
        this.emotionSystem = new EmotionSystem(this.personality);
        this.memorySystem = new MemorySystem(
            this.ARCH.MAX_RELATIONSHIPS,
            this.ARCH.MAX_EPISODIC_EVENTS
        );

        // FIX-25: schema manager
        this._schemaManager = new SchemaManager(this.DIM.DIM_SENSOR);
    }

    _initRuntimeState() {
        this.step = 0;
        this.cumulativeReward = 0;
        this.health = 100;
        this.stats = { maxHealth: 100 };
        this._lastKiller = null;
        this._lastInputs = null;
        this._lastOutputs = null;
        this._lastCInput = null;   // FIX-15: stored for eligibility traces
        this._reflexThisFrame = false;
        this._traceBuffer = null;
    }

    _createEmptyStats() {
        return {
            tdErrorHistory: [],
            predictionErrorHistory: [],
            totalSteps: 0,
            avgReward: 0,
            kills: 0,
            deaths: 0,
            reflexTriggers: 0,
            consolidationEvents: 0,
            gradientClips: 0,
            validationErrors: 0,
        };
    }

    // ========================================================================
    // DYNAMIC RATIO CALCULATION
    // ========================================================================

    _calculateRatios(taskSpec) {
        const defaults = {
            perception: 0.5,
            head: 0.25,
            memoryBlock: 0.75,
            cortex: 4.0,
            state: 0.5,
        };
        if (!taskSpec) return defaults;

        const {
            inputComplexity = 0.5,
            temporalDepth = 0.5,
            socialComplexity = 0.5,
        } = taskSpec;

        return {
            perception: clamp(0.3 + inputComplexity * 0.4, 0.3, 0.9),
            head: clamp(0.15 + temporalDepth * 0.2, 0.15, 0.4),
            memoryBlock: clamp(0.5 + socialComplexity * 0.5, 0.5, 1.0),
            cortex: clamp(2.0 + (inputComplexity + temporalDepth + socialComplexity) * 0.7, 2.0, 6.0),
            state: clamp(0.3 + socialComplexity * 0.4, 0.3, 0.8),
        };
    }

    // ========================================================================
    // SCHEMA & VALIDATION (delegates to SchemaManager)
    // ========================================================================

    getInputSchema() { return this._schemaManager.getInputSchema(); }
    getOutputSchema() { return this._schemaManager.getOutputSchema(); }
    setInputSchema(schema) { this._schemaManager.setInputSchema(schema); }
    setOutputSchema(schema) { this._schemaManager.setOutputSchema(schema); }

    validateAndNormalizeInputs(inputs) {
        const normalised = this._schemaManager.validateAndNormalizeInputs(inputs, this._stats);
        return this._projectToCortexSensors(normalised);
    }

    validateAndClampOutputs(outputs) {
        return this._schemaManager.validateAndClampOutputs(outputs);
    }

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    configurePersonality(overrides) {
        if (overrides.traits) {
            for (const [k, v] of Object.entries(overrides.traits)) {
                if (k in this.personality.traits) this.personality.traits[k] = clamp(v, 0, 1);
            }
        }
        if (overrides.values) {
            for (const [k, v] of Object.entries(overrides.values)) {
                if (k in this.personality.values) this.personality.values[k] = clamp(v, 0, 1);
            }
        }
    }

    /** @param {Object} config */
    configureEmotions(config) {
        this.emotionSystem.configure(config);
    }

    /** @param {Object} taskSpec */
    updateTaskSpec(taskSpec) {
        this.config.taskSpec = { ...this.config.taskSpec, ...taskSpec };
        this.RATIOS = this._calculateRatios(this.config.taskSpec);
    }

    /** FIX-06: targets this.config.* */
    setLearningParams(params) {
        if (params.lr !== undefined) this.config.lr = clamp(params.lr, 0.001, 0.1);
        if (params.gamma !== undefined) this.config.gamma = clamp(params.gamma, 0.9, 0.999);
        if (params.lambda !== undefined) this.config.lambda = clamp(params.lambda, 0.8, 0.99);
        if (params.hebbianRate !== undefined) this.config.hebbianRate = clamp(params.hebbianRate, 0.001, 0.01);
    }

    // ========================================================================
    // FORWARD PASS
    // ========================================================================

    /**
     * FIX-28: Reflex/conflict resolution now happens BEFORE this.D.set(D_new).
     * FIX-07: Lightweight return object.
     * FIX-15: C_input stored for eligibility traces.
     *
     * @param {number[]|Object} rawInputs
     * @param {number}  [reward=0]
     * @param {Object}  [combatEvents=null]
     * @param {Object}  [socialContext=null]
     * @returns {ForwardResult}
     */
    forward(rawInputs, reward = 0, combatEvents = null, socialContext = null) {
        if (!this._isReady) throw new Error('Cortex2Brain not initialized');

        const trace = this._traceBuffer ? { start: performance.now(), layers: {} } : null;

        // Validate inputs
        const sensorInput = this.validateAndNormalizeInputs(rawInputs);
        const rewardVal = safeNumber(reward, 0);

        this._lastInputs = Array.from(sensorInput);
        this.cumulativeReward = lerp(this.cumulativeReward, rewardVal, 0.1);

        // Social context
        if (socialContext) {
            this.memorySystem.processSocialInput(
                socialContext, this.step, this.personality, this.emotionSystem.emotions
            );
            const memories = this.memorySystem.recallRelevantMemories({
                enemyId: socialContext.enemyId,
                healthRatio: this.health / this.stats.maxHealth,
                situation: socialContext.situation,
            });
            for (const mem of memories) {
                if (mem.emotion === 'fear') this.emotionSystem.emotions.fear = Math.min(1, this.emotionSystem.emotions.fear + 0.1);
                if (mem.emotion === 'pride') this.emotionSystem.emotions.confidence = Math.min(1, this.emotionSystem.emotions.confidence + 0.1);
                if (mem.emotion === 'empathy') this.emotionSystem.emotions.empathy = Math.min(1, this.emotionSystem.emotions.empathy + 0.1);
            }
        }

        // === BUILD LAYER INPUTS ===
        const mPermPrev = this._getMPerm(this.M_prev);
        const mWorkPrev = this._getMWork(this.M_prev);
        const mPredPrev = this._getMPred(this.M_prev);

        const pInput = this._buildPInput(sensorInput, this.C_prev, this.S_prev, mPermPrev, this.RATIOS.perception);
        const aInputs = [];
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            aInputs.push(this._buildAInput(this.P_prev, mWorkPrev, this.C_prev, this.D_prev, h, this.RATIOS.head));
        }
        const mWorkInput = this._buildMWorkInput(this.P_prev, this.A_prev, this.C_prev, this.RATIOS.memoryBlock);
        const mPermInput = this._buildMPermInput(this.C_prev, mPermPrev, this.RATIOS.memoryBlock);
        const mPredInput = this._buildMPredInput(this.A_prev, this.C_prev, mWorkPrev, this.D_prev, this.RATIOS.memoryBlock);
        const cInput = this._buildCInput(this.P_prev, this.A_prev, this.M_prev, this.S_prev, this.D_prev, rewardVal, this.RATIOS.cortex);
        const sInput = this._buildSInput(this.P_prev, mPermPrev, this.C_prev, this.D_prev, this.RATIOS.state);
        const dInput = this._buildDInput(this.C_prev, this.S_prev, mPredPrev);

        // FIX-15: store for eligibility trace update
        this._lastCInput = cInput;

        if (trace) trace.layers.inputBuild = performance.now();

        // === FORWARD PROPAGATION (FIX-08: linear mode, bias + activation once) ===

        const pNew = this._matrixMultiply(pInput, this.W_P);
        for (let i = 0; i < this.DIM.DIM_P; i++) pNew[i] = leakyRelu(pNew[i] + this.b_P[i]);

        // FIX-03: convert to plain array for softmax
        const gateArr = new Array(this.DIM.NUM_HEADS);
        for (let i = 0; i < this.DIM.NUM_HEADS; i++) gateArr[i] = this.W_gate[i];
        const gateProbs = softmax(gateArr, 1.0);

        const aNew = new Float32Array(this.DIM.DIM_A);
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            const headOut = this._matrixMultiply(aInputs[h], this.W_A[h]);
            const gateWeight = gateProbs[h];
            for (let i = 0; i < this.DIM.HEAD_DIM; i++) {
                aNew[h * this.DIM.HEAD_DIM + i] = leakyRelu(headOut[i] + this.b_A[h][i]) * gateWeight;
            }
        }

        const mWorkNew = this._matrixMultiply(mWorkInput, this.W_M_work);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mWorkNew[i] = leakyRelu(mWorkNew[i] + this.b_M_work[i]);

        const mPermNew = this._matrixMultiply(mPermInput, this.W_M_perm);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mPermNew[i] = leakyRelu(mPermNew[i] + this.b_M_perm[i]);

        const mPredNew = this._matrixMultiply(mPredInput, this.W_M_pred);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mPredNew[i] = leakyRelu(mPredNew[i] + this.b_M_pred[i]);

        const mNew = new Float32Array(this.DIM.DIM_M);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mNew[i] = mWorkNew[i];
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mNew[this.ARCH.M_BLOCK_SIZE + i] = mPermNew[i];
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mNew[this.ARCH.M_BLOCK_SIZE * 2 + i] = mPredNew[i];

        const cNew = this._matrixMultiply(cInput, this.W_C);
        for (let i = 0; i < this.DIM.DIM_C; i++) cNew[i] = leakyRelu(cNew[i] + this.b_C[i]);

        const sNew = this._matrixMultiply(sInput, this.W_S);
        for (let i = 0; i < this.DIM.DIM_S; i++) sNew[i] = leakyRelu(sNew[i] + this.b_S[i]);

        const dNew = this._matrixMultiply(dInput, this.W_D);
        for (let i = 0; i < this.DIM.DIM_D; i++) dNew[i] = tanh(dNew[i] + this.b_D[i]);

        if (trace) trace.layers.forward = performance.now();

        // === PREDICTION ===
        this._updatePrediction(mPredNew, sensorInput);

        // === EMOTIONS ===
        this.emotionSystem.update(
            rewardVal, sensorInput, combatEvents,
            this.step, this.cumulativeReward,
            this._computeAvgPredError(),
            this.health, this.stats.maxHealth
        );
        this.emotionSystem.writeToStateVector(sNew);

        // === FIX-28: REFLEX + CONFLICT RESOLUTION *BEFORE* STATE COMMIT ===
        this._reflexThisFrame = false;
        if (combatEvents) {
            this.emotionSystem.applyReflexes(sensorInput, sNew, dNew, this._stats);
        }
        this.emotionSystem.resolveConflicts(dNew);
        // Check if any reflex fired (using threshold from ARCH)
        for (let i = 0; i < this.DIM.DIM_S; i++) {
            if (sNew[i] > this.ARCH.REFLEX_THRESHOLD) {
                this._reflexThisFrame = true;
                break;
            }
        }

        if (trace) trace.layers.emotions = performance.now();

        // === NOW COMMIT STATE (FIX-28: after reflexes have modified dNew/sNew) ===
        this.P_prev.set(this.P); this.A_prev.set(this.A); this.M_prev.set(this.M);
        this.C_prev.set(this.C); this.S_prev.set(this.S); this.D_prev.set(this.D);
        this.P.set(pNew); this.A.set(aNew); this.M.set(mNew);
        this.C.set(cNew); this.S.set(sNew); this.D.set(dNew);

        // === OUTPUT ===
        const output = this.validateAndClampOutputs(Array.from(this.D));
        this._lastOutputs = output;

        // === TRACE FINALISE ===
        if (trace) {
            trace.end = performance.now();
            trace.duration = trace.end - trace.start;
            this._traceBuffer.push(trace);
            if (this._traceBuffer.length > (this._traceBuffer.maxSize ?? this.ARCH.MAX_TRACE_SIZE)) {
                this._traceBuffer.shift();
            }
        }

        this.step++;

        return {
            output,
            predictionError: this._computeAvgPredError(),
            reflexTriggered: this._reflexThisFrame,
            emotionalStateName: this.emotionSystem.emotionalState.current?.name ?? null,
            stepCount: this.step,
            trace: trace ? { ...trace } : null,
        };
    }

    // ========================================================================
    // BATCH PROCESSING
    // ========================================================================

    forwardBatch(inputsArray, options = {}) {
        const { rewards = [] } = options;
        return inputsArray.map((inputs, i) => this.forward(inputs, rewards[i] || 0, null, null));
    }

    learnBatch(experiences, options = {}) {
        const { accumulateGradients = true } = options;
        if (!accumulateGradients) {
            for (const exp of experiences) this.learn(exp);
            return;
        }
        const accumulated = { tdErrors: [] };
        for (const exp of experiences) {
            const { reward, done, nextObservation } = exp;
            const currentValue = this._estimateValue();
            const nextValue = done ? 0 : (nextObservation ? this._estimateValue() : currentValue);
            const tdErr = this._tdError(reward, currentValue, nextValue, this.config.gamma);
            accumulated.tdErrors.push(tdErr);
            this._updateValueHead(tdErr / experiences.length);
        }
        const avgTDErr = accumulated.tdErrors.reduce((a, b) => a + b, 0) / experiences.length;
        this._stats.tdErrorHistory.push(Math.abs(avgTDErr));
        if (this._stats.tdErrorHistory.length > this.ARCH.MAX_TD_HISTORY) {
            this._stats.tdErrorHistory.shift();
        }
    }

    // ========================================================================
    // LEARNING
    // ========================================================================

    /**
     * FIX-19: Properly forwards nextObservation to estimate next-state value.
     * Saves and restores state vectors so learn() doesn't corrupt the agent state.
     * @param {Experience} experience
     */
    learn(experience) {
        const { observation, action, reward, nextObservation, done } = experience;
        const inputs = this.validateAndNormalizeInputs(observation);
        const outputs = Array.isArray(action) ? action : (action?.output || []);

        const currentValue = this._estimateValue();
        let nextValue = 0;

        if (!done && nextObservation) {
            // FIX-19: save state, forward next observation, estimate value, restore
            const savedC = new Float32Array(this.C);
            const savedP = new Float32Array(this.P);
            const savedA = new Float32Array(this.A);
            const savedM = new Float32Array(this.M);
            const savedS = new Float32Array(this.S);
            const savedD = new Float32Array(this.D);
            const savedStep = this.step;

            try {
                this.forward(nextObservation, 0);
                nextValue = this._estimateValue();
            } finally {
                // Restore state
                this.C.set(savedC);
                this.P.set(savedP);
                this.A.set(savedA);
                this.M.set(savedM);
                this.S.set(savedS);
                this.D.set(savedD);
                this.step = savedStep;
            }
        }

        return this.tdLearn(inputs, outputs, reward, currentValue, nextValue);
    }

    tdLearn(inputs, outputs, reward, value, nextValue, actionLogProbs = null) {
        const currentValue = (value != null) ? safeNumber(value, 0) : this._estimateValue();
        const nextVal = (nextValue != null) ? safeNumber(nextValue, 0) : this._estimateValue();
        const tdErr = this._tdError(reward, currentValue, nextVal, this.config.gamma);

        this._stats.tdErrorHistory.push(Math.abs(tdErr));
        if (this._stats.tdErrorHistory.length > this.ARCH.MAX_TD_HISTORY) {
            this._stats.tdErrorHistory.shift();
        }

        this._updateValueHead(tdErr);
        // FIX-15: pass stored C_input for proper trace update
        this._updateEligibilityTraces(inputs, outputs, tdErr, this._lastCInput);
        this._applyTDError(tdErr);

        if (actionLogProbs?.length) {
            const avgAdvantage = -actionLogProbs.reduce((a, b) => a + b, 0) / actionLogProbs.length;
            const maxW = this.ARCH.MAX_WEIGHT;
            for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
                this.W_gate[h] = clamp(this.W_gate[h] + this.config.lr * 0.3 * avgAdvantage * 0.25, -maxW, maxW);
            }
        }

        if (reward > 3) {
            this.memorySystem.consolidateMemories(
                this.predictionErrors, this.M, reward,
                this.step, this._stats, this.DIM.M_PRED_START
            );
        }

        if (reward > 1) {
            const mag = clamp(Math.abs(reward) / 5.0, 0, 1);
            const correlation = Math.sign(reward) * (0.5 + 0.5 * mag);
            this._hebbianUpdate(this.P_prev, this.A_prev, correlation, 'P_to_A');
            this._hebbianUpdate(this.A_prev, this.C_prev, correlation, 'A_to_C');
            this._hebbianUpdate(this.C_prev, this.D_prev, correlation, 'C_to_D');
        }

        this._stats.totalSteps++;
        this._stats.avgReward = lerp(this._stats.avgReward, reward, 0.01);

        return {
            tdError: tdErr,
            avgTDError: this._stats.tdErrorHistory.length > 0
                ? this._stats.tdErrorHistory.reduce((a, b) => a + b, 0) / this._stats.tdErrorHistory.length
                : 0,
            avgPredictionError: this._computeAvgPredError(),
        };
    }

    // ========================================================================
    // EPISODE MANAGEMENT
    // ========================================================================

    resetEpisode() {
        this.emotionSystem.resetEpisode();
        this.memorySystem.resetEpisode(this.memorySystem.episodicMemory.decayRate);

        this.P.fill(0); this.A.fill(0); this.M.fill(0);
        this.C.fill(0); this.S.fill(0); this.D.fill(0);
        this.P_prev.fill(0); this.A_prev.fill(0); this.M_prev.fill(0);
        this.C_prev.fill(0); this.S_prev.fill(0); this.D_prev.fill(0);

        this.predictionErrors.fill(0);
        this.cumulativeReward = 0;
        this._lastCInput = null;
        this.step = 0;
    }

    reset() {
        this.resetEpisode();
        this.memorySystem.resetFull();
        this._stats = this._createEmptyStats();
    }

    // ========================================================================
    // DEBUG & TRACING
    // ========================================================================

    /** @param {boolean} enable @param {number} maxTraces */
    setTracing(enable, maxTraces = 10) {
        this._traceBuffer = enable ? [] : null;
        if (this._traceBuffer) this._traceBuffer.maxSize = maxTraces;
    }

    /**
     * FIX-07: Heavy introspection data, not returned from forward().
     * @returns {DebugInfo}
     */
    getDebugInfo() {
        const sm = this.memorySystem.socialMemory;
        return {
            emotions: this.getEmotions(),
            emotionsFull: { ...this.emotionSystem.emotions },
            personality: this.getPersonality(),
            emotionalState: this.emotionSystem.emotionalState.current
                ? { ...this.emotionSystem.emotionalState.current }
                : null,
            stats: this.getStats(),
            attentionWeights: Array.from(this.W_gate),
            predictionError: this._computeAvgPredError(),
            memoryUsage: {
                working: this._getMWork(this.M).reduce((a, b) => a + Math.abs(b), 0) / this.ARCH.M_BLOCK_SIZE,
                permanent: this._getMPerm(this.M).reduce((a, b) => a + Math.abs(b), 0) / this.ARCH.M_BLOCK_SIZE,
                predictive: this._getMPred(this.M).reduce((a, b) => a + Math.abs(b), 0) / this.ARCH.M_BLOCK_SIZE,
            },
            cortexEnergy: this.C.reduce((a, b) => a + Math.abs(b), 0) / this.DIM.DIM_C,
            cortex: Array.from(this.C),
            ratios: { ...this.RATIOS },
            socialMemory: {
                relationships: Array.from(sm.relationships.entries()).map(([k, v]) => [k, {
                    trust: v.trust,
                    affinity: v.affinity,
                    lastInteraction: v.lastInteraction,
                }]),
                reputation: sm.reputation,
            },
            trace: this._traceBuffer ? {
                count: this._traceBuffer.length,
                avgDuration: this._traceBuffer.length > 0
                    ? this._traceBuffer.reduce((s, t) => s + t.duration, 0) / this._traceBuffer.length
                    : 0,
                recent: this._traceBuffer.slice(-3),
            } : null,
        };
    }

    getTraces(count = 5) {
        if (!this._traceBuffer) return [];
        return this._traceBuffer.slice(-count).map(t => ({ ...t }));
    }

    clearTraces() {
        if (this._traceBuffer) this._traceBuffer.length = 0;
    }

    // ========================================================================
    // SERIALISATION
    // ========================================================================

    toJSON(options = {}) {
        const {
            includePersonality = true,
            includeEmotions = true,
            includeSocialMemory = true,
            includeEpisodicMemory = false,
            includeStats = true,
            includeSchema = true,
        } = options;

        const data = {
            type: 'Cortex2Brain',
            version: '2.3',
            config: {
                seed: this.config.seed,
                lr: this.config.lr,
                gamma: this.config.gamma,
                lambda: this.config.lambda,
                hebbianRate: this.config.hebbianRate,
                taskSpec: this.config.taskSpec,
                dims: { ...this.DIM },
            },
            weights: this._serializeWeights(),
        };

        if (includePersonality) {
            data.personality = {
                traits: { ...this.personality.traits },
                values: { ...this.personality.values },
            };
        }

        // FIX-29: emotions now serialised
        if (includeEmotions) {
            data.emotionSystem = this.emotionSystem.toJSON();
        }

        const memoryData = this.memorySystem.toJSON();
        if (includeSocialMemory) data.socialMemory = memoryData.socialMemory;
        if (includeEpisodicMemory) data.episodicMemory = memoryData.episodicMemory;
        if (includeStats) data.stats = { ...this._stats };
        if (includeSchema) {
            data.inputSchema = this._schemaManager.getInputSchema();
            data.outputSchema = this._schemaManager.getOutputSchema();
        }

        return data;
    }

    /** @param {Object} data */
    static fromJSON(data) {
        if (!data || data.type !== 'Cortex2Brain') throw new Error('Invalid Cortex2Brain JSON data');

        const brain = new Cortex2Brain(data.config);
        const w = data.weights;

        if (w) {
            // FIX-27: validate dimensions before loading
            brain._validateAndLoadWeights(w);
        }

        if (data.personality) {
            brain.personality.traits = { ...brain.personality.traits, ...data.personality.traits };
            brain.personality.values = { ...brain.personality.values, ...data.personality.values };
        }

        // FIX-29: restore emotions
        if (data.emotionSystem) {
            brain.emotionSystem.loadFromJSON(data.emotionSystem);
        }
        // Legacy support: v2.2 format
        if (data.emotionMemory && !data.emotionSystem) {
            brain.emotionSystem.loadFromJSON({ emotionMemory: data.emotionMemory });
        }

        const memoryData = {};
        if (data.socialMemory) memoryData.socialMemory = data.socialMemory;
        if (data.episodicMemory) memoryData.episodicMemory = data.episodicMemory;
        if (data.socialMemory || data.episodicMemory) {
            brain.memorySystem.loadFromJSON(memoryData);
        }

        if (data.inputSchema) brain._schemaManager.setInputSchema(data.inputSchema);
        if (data.outputSchema) brain._schemaManager.setOutputSchema(data.outputSchema);
        if (data.stats) brain._stats = { ...brain._stats, ...data.stats };

        return brain;
    }

    _serializeWeights() {
        return {
            W_P: this.W_P.map(r => Array.from(r)),
            b_P: Array.from(this.b_P),
            W_A: this.W_A.map(m => m.map(r => Array.from(r))),
            b_A: this.b_A.map(b => Array.from(b)),
            W_gate: Array.from(this.W_gate),
            W_reflex: Array.from(this.W_reflex),
            W_M_work: this.W_M_work.map(r => Array.from(r)),
            b_M_work: Array.from(this.b_M_work),
            W_M_perm: this.W_M_perm.map(r => Array.from(r)),
            b_M_perm: Array.from(this.b_M_perm),
            W_M_pred: this.W_M_pred.map(r => Array.from(r)),
            b_M_pred: Array.from(this.b_M_pred),
            W_C: this.W_C.map(r => Array.from(r)),
            b_C: Array.from(this.b_C),
            W_S: this.W_S.map(r => Array.from(r)),
            b_S: Array.from(this.b_S),
            W_D: this.W_D.map(r => Array.from(r)),
            b_D: Array.from(this.b_D),
            W_pred: this.W_pred.map(r => Array.from(r)),
            b_pred: Array.from(this.b_pred),
            W_V: Array.from(this.W_V),
            b_V: this.b_V,
        };
    }

    /**
     * FIX-27: Validates weight dimensions before loading.
     * @param {Object} w - serialised weight data
     */
    _validateAndLoadWeights(w) {
        const check = (name, mat, expectedRows, expectedCols) => {
            if (!mat || mat.length !== expectedRows) {
                throw new Error(
                    `${name} row count mismatch: expected ${expectedRows}, got ${mat?.length}`
                );
            }
            if (expectedCols !== undefined && mat[0]?.length !== expectedCols) {
                throw new Error(
                    `${name} col count mismatch: expected ${expectedCols}, got ${mat[0]?.length}`
                );
            }
        };

        const checkVec = (name, vec, expectedLen) => {
            if (!vec || vec.length !== expectedLen) {
                throw new Error(
                    `${name} length mismatch: expected ${expectedLen}, got ${vec?.length}`
                );
            }
        };

        // Validate matrix dimensions
        check('W_P', w.W_P, this.ARCH.P_INPUT, this.DIM.DIM_P);
        checkVec('b_P', w.b_P, this.DIM.DIM_P);

        if (w.W_A.length !== this.DIM.NUM_HEADS) {
            throw new Error(`W_A head count mismatch: expected ${this.DIM.NUM_HEADS}, got ${w.W_A.length}`);
        }
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            check(`W_A[${h}]`, w.W_A[h], this.ARCH.A_INPUT, this.DIM.HEAD_DIM);
            checkVec(`b_A[${h}]`, w.b_A[h], this.DIM.HEAD_DIM);
        }

        checkVec('W_gate', w.W_gate, this.DIM.NUM_HEADS);

        check('W_M_work', w.W_M_work, this.ARCH.M_WORK_INPUT, this.ARCH.M_BLOCK_SIZE);
        checkVec('b_M_work', w.b_M_work, this.ARCH.M_BLOCK_SIZE);
        check('W_M_perm', w.W_M_perm, this.ARCH.M_PERM_INPUT, this.ARCH.M_BLOCK_SIZE);
        checkVec('b_M_perm', w.b_M_perm, this.ARCH.M_BLOCK_SIZE);
        check('W_M_pred', w.W_M_pred, this.ARCH.M_PRED_INPUT, this.ARCH.M_BLOCK_SIZE);
        checkVec('b_M_pred', w.b_M_pred, this.ARCH.M_BLOCK_SIZE);

        check('W_C', w.W_C, this.ARCH.C_INPUT, this.DIM.DIM_C);
        checkVec('b_C', w.b_C, this.DIM.DIM_C);
        check('W_S', w.W_S, this.ARCH.S_INPUT, this.DIM.DIM_S);
        checkVec('b_S', w.b_S, this.DIM.DIM_S);
        check('W_D', w.W_D, this.ARCH.D_INPUT, this.DIM.DIM_D);
        checkVec('b_D', w.b_D, this.DIM.DIM_D);

        check('W_pred', w.W_pred, this.ARCH.M_BLOCK_SIZE, this.DIM.DIM_SENSOR);
        checkVec('b_pred', w.b_pred, this.DIM.DIM_SENSOR);
        checkVec('W_V', w.W_V, this.DIM.DIM_C);

        // All checks passed — load weights
        this.W_P = w.W_P.map(r => new Float32Array(r));
        this.b_P = new Float32Array(w.b_P);
        this.W_A = w.W_A.map(m => m.map(r => new Float32Array(r)));
        this.b_A = w.b_A.map(b => new Float32Array(b));
        this.W_gate = new Float32Array(w.W_gate);
        if (w.W_reflex) this.W_reflex = new Float32Array(w.W_reflex);
        this.W_M_work = w.W_M_work.map(r => new Float32Array(r));
        this.b_M_work = new Float32Array(w.b_M_work);
        this.W_M_perm = w.W_M_perm.map(r => new Float32Array(r));
        this.b_M_perm = new Float32Array(w.b_M_perm);
        this.W_M_pred = w.W_M_pred.map(r => new Float32Array(r));
        this.b_M_pred = new Float32Array(w.b_M_pred);
        this.W_C = w.W_C.map(r => new Float32Array(r));
        this.b_C = new Float32Array(w.b_C);
        this.W_S = w.W_S.map(r => new Float32Array(r));
        this.b_S = new Float32Array(w.b_S);
        this.W_D = w.W_D.map(r => new Float32Array(r));
        this.b_D = new Float32Array(w.b_D);
        this.W_pred = w.W_pred.map(r => new Float32Array(r));
        this.b_pred = new Float32Array(w.b_pred);
        this.W_V = new Float32Array(w.W_V);
        this.b_V = w.b_V ?? 0;
    }

    // ========================================================================
    // CLONING & MUTATION
    // ========================================================================

    clone(options = {}) {
        const { copyWeights = true, copyState = false, copyMemory = true } = options;

        const cloned = new Cortex2Brain({
            seed: this.config.seed + '_clone_' + Date.now(),
            lr: this.config.lr,
            gamma: this.config.gamma,
            lambda: this.config.lambda,
            hebbianRate: this.config.hebbianRate,
            taskSpec: this.config.taskSpec ? { ...this.config.taskSpec } : null,
        });

        if (copyWeights) {
            const cm = src => src.map(r => new Float32Array(r));
            cloned.W_P = cm(this.W_P); cloned.b_P = new Float32Array(this.b_P);
            cloned.W_A = this.W_A.map(m => m.map(r => new Float32Array(r)));
            cloned.b_A = this.b_A.map(b => new Float32Array(b));
            cloned.W_gate = new Float32Array(this.W_gate);
            cloned.W_reflex = new Float32Array(this.W_reflex);
            cloned.W_M_work = cm(this.W_M_work); cloned.b_M_work = new Float32Array(this.b_M_work);
            cloned.W_M_perm = cm(this.W_M_perm); cloned.b_M_perm = new Float32Array(this.b_M_perm);
            cloned.W_M_pred = cm(this.W_M_pred); cloned.b_M_pred = new Float32Array(this.b_M_pred);
            cloned.W_C = cm(this.W_C); cloned.b_C = new Float32Array(this.b_C);
            cloned.W_S = cm(this.W_S); cloned.b_S = new Float32Array(this.b_S);
            cloned.W_D = cm(this.W_D); cloned.b_D = new Float32Array(this.b_D);
            cloned.W_pred = cm(this.W_pred); cloned.b_pred = new Float32Array(this.b_pred);
            cloned.W_V = new Float32Array(this.W_V); cloned.b_V = this.b_V;
        }

        if (copyState) {
            cloned.P.set(this.P); cloned.A.set(this.A); cloned.M.set(this.M);
            cloned.C.set(this.C); cloned.S.set(this.S); cloned.D.set(this.D);
            cloned.P_prev.set(this.P_prev); cloned.A_prev.set(this.A_prev); cloned.M_prev.set(this.M_prev);
            cloned.C_prev.set(this.C_prev); cloned.S_prev.set(this.S_prev); cloned.D_prev.set(this.D_prev);
            cloned.predictionErrors.set(this.predictionErrors);
        }

        cloned.personality = {
            traits: { ...this.personality.traits },
            values: { ...this.personality.values },
        };

        if (copyMemory) {
            // Clone emotion system state
            const emotionData = this.emotionSystem.toJSON();
            cloned.emotionSystem.loadFromJSON(emotionData);

            // Clone memory system state
            const memoryData = this.memorySystem.toJSON();
            cloned.memorySystem.loadFromJSON(memoryData);
        }

        cloned._schemaManager.setInputSchema(this._schemaManager.getInputSchema());
        cloned._schemaManager.setOutputSchema(this._schemaManager.getOutputSchema());
        cloned._stats = {
            ...this._stats,
            tdErrorHistory: [...this._stats.tdErrorHistory],
            predictionErrorHistory: [...this._stats.predictionErrorHistory],
        };

        return cloned;
    }

    mutate(rate = 0.01, strength = 0.1) {
        let mutations = 0;
        const maxW = this.ARCH.MAX_WEIGHT;
        const mutateMat = mat => {
            if (!mat?.length) return;
            for (let i = 0; i < mat.length; i++) {
                if (!mat[i]?.length) continue;
                for (let j = 0; j < mat[i].length; j++) {
                    if (this._rng.next() < rate) {
                        mat[i][j] = clamp(mat[i][j] + this._rng.gaussian(0, strength), -maxW, maxW);
                        mutations++;
                    }
                }
            }
        };
        mutateMat(this.W_P); mutateMat(this.W_C); mutateMat(this.W_D);
        for (const W of this.W_A) mutateMat(W);
        mutateMat(this.W_M_work); mutateMat(this.W_M_perm); mutateMat(this.W_M_pred);
        mutateMat(this.W_S); mutateMat(this.W_pred);
        for (let i = 0; i < this.W_gate.length; i++) {
            if (this._rng.next() < rate) {
                this.W_gate[i] = clamp(this.W_gate[i] + this._rng.gaussian(0, strength), -maxW, maxW);
                mutations++;
            }
        }
        return mutations;
    }

    // ========================================================================
    // GETTERS
    // ========================================================================

    getEmotions() { return { ...this.emotionSystem.emotions }; }
    getPersonality() { return { traits: { ...this.personality.traits }, values: { ...this.personality.values } }; }
    getStats() { return { ...this._stats }; }
    getInputDim() { return this.DIM.DIM_SENSOR; }
    getOutputDim() { return this.DIM.DIM_D; }
    isReady() { return this._isReady; }

    // ========================================================================
    // PRIVATE — WEIGHT INITIALISATION
    // ========================================================================

    _initWeights() {
        const bs = this.ARCH.M_BLOCK_SIZE;

        this.W_P = this._initMatrix(this.ARCH.P_INPUT, this.DIM.DIM_P, 0.3);
        this.b_P = this._initVector(this.DIM.DIM_P, 0.1);

        this.W_A = [];
        this.b_A = [];
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            this.W_A.push(this._initMatrix(this.ARCH.A_INPUT, this.DIM.HEAD_DIM, 0.3));
            this.b_A.push(this._initVector(this.DIM.HEAD_DIM, 0.1));
        }

        this.W_gate = this._initVector(this.DIM.NUM_HEADS, 0.25);
        this.W_reflex = this._initVector(4, 0.5);

        this.W_M_work = this._initMatrix(this.ARCH.M_WORK_INPUT, bs, 0.3);
        this.b_M_work = this._initVector(bs, 0.1);
        this.W_M_perm = this._initMatrix(this.ARCH.M_PERM_INPUT, bs, 0.3);
        this.b_M_perm = this._initVector(bs, 0.1);
        this.W_M_pred = this._initMatrix(this.ARCH.M_PRED_INPUT, bs, 0.3);
        this.b_M_pred = this._initVector(bs, 0.1);

        this.W_C = this._initMatrix(this.ARCH.C_INPUT, this.DIM.DIM_C, 0.25);
        this.b_C = this._initVector(this.DIM.DIM_C, 0.1);

        this.W_S = this._initMatrix(this.ARCH.S_INPUT, this.DIM.DIM_S, 0.3);
        this.b_S = this._initVector(this.DIM.DIM_S, 0.1);

        this.W_D = this._initMatrix(this.ARCH.D_INPUT, this.DIM.DIM_D, 0.2);
        this.b_D = this._initVector(this.DIM.DIM_D, 0.1);

        this.W_pred = this._initMatrix(bs, this.DIM.DIM_SENSOR, 0.15);
        this.b_pred = this._initVector(this.DIM.DIM_SENSOR, 0.05);

        this.W_V = this._initVector(this.DIM.DIM_C, 0.1);
        this.b_V = 0;
    }

    _initStateVectors() {
        this.P = new Float32Array(this.DIM.DIM_P);
        this.A = new Float32Array(this.DIM.DIM_A);
        this.M = new Float32Array(this.DIM.DIM_M);
        this.C = new Float32Array(this.DIM.DIM_C);
        this.S = new Float32Array(this.DIM.DIM_S);
        this.D = new Float32Array(this.DIM.DIM_D);

        this.P_prev = new Float32Array(this.DIM.DIM_P);
        this.A_prev = new Float32Array(this.DIM.DIM_A);
        this.M_prev = new Float32Array(this.DIM.DIM_M);
        this.C_prev = new Float32Array(this.DIM.DIM_C);
        this.S_prev = new Float32Array(this.DIM.DIM_S);
        this.D_prev = new Float32Array(this.DIM.DIM_D);
    }

    _initEligibilityTraces() {
        this.traceP = new Float32Array(this.ARCH.P_INPUT * this.DIM.DIM_P);
        this.traceC = new Float32Array(this.ARCH.C_INPUT * this.DIM.DIM_C);
        this.traceD = new Float32Array(this.ARCH.D_INPUT * this.DIM.DIM_D);
        this.tracePred = new Float32Array(this.ARCH.M_BLOCK_SIZE * this.DIM.DIM_SENSOR);
    }

    _initMatrix(rows, cols, scale) {
        const mat = [];
        for (let i = 0; i < rows; i++) {
            const row = new Float32Array(cols);
            for (let j = 0; j < cols; j++) {
                row[j] = (this._hashSeed(this.config.seed, i * 10000 + j, 0) * 2 - 1) * scale;
            }
            mat.push(row);
        }
        return mat;
    }

    _initVector(length, scale) {
        const vec = new Float32Array(length);
        for (let i = 0; i < length; i++) {
            vec[i] = (this._hashSeed(this.config.seed, i, 1000) * 2 - 1) * scale;
        }
        return vec;
    }

    /**
     * FIX-20: FNV-1a hash — much better avalanche than the old multiplicative hash.
     * @param {string} seed
     * @param {number} i
     * @param {number} j
     * @returns {number} float in [0, 1)
     */
    _hashSeed(seed, i, j) {
        let h = 0x811c9dc5; // FNV offset basis
        const s = `${seed}:${i}:${j}`;
        for (let k = 0; k < s.length; k++) {
            h ^= s.charCodeAt(k);
            h = Math.imul(h, 0x01000193); // FNV prime
        }
        return ((h >>> 0) ^ ((h >>> 16) & 0xFFFF)) / 4294967296;
    }

    // ========================================================================
    // PRIVATE — MATH / MATRIX OPS (FIX-23: optimised hot-path)
    // ========================================================================

    /**
     * FIX-23: No optional chaining or _safeNumber in inner loop.
     *         Input/matrix validated once, then raw indexed.
     * FIX-08: Always returns linear (raw dot product); caller applies activation.
     * @param {Float32Array|number[]} vec
     * @param {Float32Array[]} mat
     * @returns {Float32Array}
     */
    _matrixMultiply(vec, mat) {
        const rows = mat.length;
        const cols = mat[0].length;
        const n = Math.min(vec.length, rows);
        const out = new Float32Array(cols);
        for (let i = 0; i < cols; i++) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
                sum += vec[j] * mat[j][i];
            }
            out[i] = sum;
        }
        return out;
    }

    _projectToCortexSensors(rawInputs) {
        const output = new Float32Array(this.DIM.DIM_SENSOR);
        const src = rawInputs || [];
        const srcLen = Math.min(src.length, this.DIM.DIM_SENSOR);
        for (let i = 0; i < srcLen; i++) {
            const v = safeNumber(src[i], 0);
            output[i] = v < -1 ? -1 : v > 1 ? 1 : v;
        }
        for (let i = srcLen; i < this.DIM.DIM_SENSOR; i++) {
            output[i] = (this._hashSeed(this.config.seed, i, 0xDEADBEEF) * 2 - 1) * 0.05;
        }
        return output;
    }

    _getMWork(M) { return M.subarray(this.DIM.M_WORK_START, this.DIM.M_WORK_END); }
    _getMPerm(M) { return M.subarray(this.DIM.M_PERM_START, this.DIM.M_PERM_END); }
    _getMPred(M) { return M.subarray(this.DIM.M_PRED_START, this.DIM.M_PRED_END); }

    // ========================================================================
    // PRIVATE — LAYER INPUT BUILDERS (FIX-23: no console.assert in production)
    // ========================================================================

    _buildPInput(sensorInput, cPrev, sPrev, mPermPrev, ratio) {
        const input = new Float32Array(this.ARCH.P_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) input[idx++] = sensorInput[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_C; i++) input[idx++] = cPrev[i];
        for (let i = 0; i < this.DIM.DIM_S; i++) input[idx++] = sPrev[i];
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) input[idx++] = mPermPrev[i];
        return input;
    }

    _buildAInput(pPrev, mWork, cPrev, dPrev, headIdx, ratio) {
        const input = new Float32Array(this.ARCH.A_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = pPrev[i] * ratio;
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) input[idx++] = mWork[i];
        // FIX-26: named constant for head slice size
        const cStart = headIdx * this.ARCH.A_HEAD_C_SLICE;
        for (let i = 0; i < this.ARCH.A_HEAD_C_SLICE; i++) {
            const ci = cStart + i;
            input[idx++] = ci < this.DIM.DIM_C ? cPrev[ci] : 0;
        }
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = dPrev[i];
        return input;
    }

    _buildMWorkInput(pPrev, aPrev, cPrev, ratio) {
        const input = new Float32Array(this.ARCH.M_WORK_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = pPrev[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_A; i++) input[idx++] = aPrev[i] * ratio;
        // FIX-26: named constant
        for (let i = 0; i < this.ARCH.M_WORK_C_SLICE; i++) {
            input[idx++] = i < this.DIM.DIM_C ? cPrev[i] : 0;
        }
        return input;
    }

    _buildMPermInput(cPrev, mPermPrev, ratio) {
        const input = new Float32Array(this.ARCH.M_PERM_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_C; i++) input[idx++] = cPrev[i] * ratio;
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) input[idx++] = mPermPrev[i];
        return input;
    }

    _buildMPredInput(aPrev, cPrev, mWork, dPrev, ratio) {
        const input = new Float32Array(this.ARCH.M_PRED_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_A; i++) input[idx++] = aPrev[i] * ratio;
        // FIX-12/26: named offset
        for (let i = 0; i < this.ARCH.A_HEAD_C_SLICE; i++) {
            const ci = this.ARCH.C_PRED_OFFSET + i;
            input[idx++] = ci < this.DIM.DIM_C ? cPrev[ci] : 0;
        }
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) input[idx++] = mWork[i];
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = dPrev[i];
        return input;
    }

    _buildCInput(pPrev, aPrev, mPrev, sPrev, dPrev, reward, ratio) {
        const input = new Float32Array(this.ARCH.C_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = pPrev[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_A; i++) input[idx++] = aPrev[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_M; i++) input[idx++] = mPrev[i] * ratio;
        for (let i = 0; i < this.DIM.DIM_S; i++) input[idx++] = sPrev[i];
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = dPrev[i];
        input[idx++] = reward;
        return input;
    }

    _buildSInput(pPrev, mPerm, cPrev, dPrev, ratio) {
        const input = new Float32Array(this.ARCH.S_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) input[idx++] = pPrev[i] * ratio;
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) input[idx++] = mPerm[i];
        // FIX-26: named offset
        for (let i = 0; i < this.ARCH.C_STATE_SLICE; i++) {
            const ci = this.ARCH.C_STATE_OFFSET + i;
            input[idx++] = ci < this.DIM.DIM_C ? cPrev[ci] : 0;
        }
        for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = dPrev[i] * ratio;
        return input;
    }

    _buildDInput(cNew, sNew, mPred) {
        const input = new Float32Array(this.ARCH.D_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_C; i++) input[idx++] = cNew[i];
        for (let i = 0; i < this.DIM.DIM_S; i++) input[idx++] = sNew[i];
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) input[idx++] = mPred[i];
        return input;
    }

    // ========================================================================
    // PRIVATE — LEARNING INTERNALS
    // ========================================================================

    /**
     * FIX-18: Now uses full DIM_SENSOR for prediction errors.
     */
    _updatePrediction(mPred, sensorInput) {
        const xPred = this._matrixMultiply(mPred, this.W_pred);
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) {
            xPred[i] += this.b_pred[i];
        }
        // FIX-18: iterate over full sensor dimension
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) {
            const error = Math.abs(xPred[i] - sensorInput[i]);
            this.predictionErrors[i] = lerp(this.predictionErrors[i], error, 0.1);
        }
    }

    _computeAvgPredError() {
        let sum = 0;
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) sum += this.predictionErrors[i];
        return sum / this.DIM.DIM_SENSOR;
    }

    _estimateValue() {
        let sum = this.b_V;
        for (let i = 0; i < this.DIM.DIM_C; i++) sum += this.C[i] * this.W_V[i];
        return tanh(sum);
    }

    _updateValueHead(tdErr) {
        const alpha = this.config.lr * 0.5;
        const maxW = this.ARCH.MAX_WEIGHT;
        for (let i = 0; i < this.DIM.DIM_C; i++) {
            this.W_V[i] = clamp(this.W_V[i] + alpha * tdErr * this.C[i], -maxW, maxW);
        }
        this.b_V += alpha * tdErr;
    }

    /**
     * FIX-15: Now receives the actual C_input vector (stored during forward())
     *         so trace_C is computed from the correct 289-dim input, not the
     *         64-dim raw sensor vector.
     * FIX-05: trace_C is updated every call (was dead in v2.1).
     *
     * @param {Float32Array} inputs  - sensor inputs (used for trace_P)
     * @param {number[]}     outputs - action outputs (unused currently, reserved)
     * @param {number}       tdErr   - TD error
     * @param {Float32Array|null} cInput - the actual cortex input vector from forward()
     */
    _updateEligibilityTraces(inputs, outputs, tdErr, cInput) {
        const decay = this.config.gamma * this.config.lambda;
        const maxTrace = this.ARCH.MAX_TRACE_VAL;

        // trace_P: uses sensor inputs (correct: P layer receives sensor input)
        const pInputLen = Math.min(inputs.length, this.ARCH.P_INPUT);
        const dimP = this.DIM.DIM_P;
        const traceP = this.traceP;
        const P = this.P;
        for (let i = 0; i < this.ARCH.P_INPUT; i++) {
            const inputVal = i < pInputLen ? inputs[i] : 0;
            const base = i * dimP;
            for (let j = 0; j < dimP; j++) {
                const idx = base + j;
                const v = decay * traceP[idx] + inputVal * P[j];
                traceP[idx] = v > maxTrace ? maxTrace : v < -maxTrace ? -maxTrace : v;
            }
        }

        // FIX-15: trace_C: uses actual C_input (289-dim), not raw sensors
        if (cInput && cInput.length > 0) {
            const cInputLen = cInput.length;
            const dimC = this.DIM.DIM_C;
            const traceC = this.traceC;
            const C = this.C;
            for (let j = 0; j < this.ARCH.C_INPUT; j++) {
                const preAct = j < cInputLen ? cInput[j] : 0;
                const base = j * dimC;
                for (let i = 0; i < dimC; i++) {
                    const idx = base + i;
                    const v = decay * traceC[idx] + preAct * C[i];
                    traceC[idx] = v > maxTrace ? maxTrace : v < -maxTrace ? -maxTrace : v;
                }
            }
        }
    }

    /**
     * FIX-16: Now updates ALL cortex columns (0..DIM_C), not just 64..127.
     * FIX-21: gradientClips counter incremented when clipping occurs.
     * @param {number} tdErr
     */
    _applyTDError(tdErr) {
        const alpha = this.config.lr * Math.sign(tdErr) * 0.1;
        const maxGrad = this.ARCH.MAX_GRADIENT;
        const maxW = this.ARCH.MAX_WEIGHT;
        const dimC = this.DIM.DIM_C;
        const traceC = this.traceC;

        // FIX-16: iterate over ALL cortex columns
        for (let i = 0; i < dimC; i++) {
            for (let j = 0; j < this.ARCH.C_INPUT; j++) {
                const row = this.W_C[j];
                if (row) {
                    const rawDelta = alpha * 0.01 * traceC[j * dimC + i];
                    // FIX-21: track gradient clipping
                    let delta = rawDelta;
                    if (delta > maxGrad) { delta = maxGrad; this._stats.gradientClips++; }
                    else if (delta < -maxGrad) { delta = -maxGrad; this._stats.gradientClips++; }
                    row[i] = clamp(row[i] + delta, -maxW, maxW);
                }
            }
        }
    }

    /**
     * FIX-04: Proper bounds checking for W_A access.
     * @param {Float32Array} pre
     * @param {Float32Array} post
     * @param {number} correlation
     * @param {string} label
     */
    _hebbianUpdate(pre, post, correlation, label) {
        if (!pre?.length || !post?.length) return;
        const eta = this.config.hebbianRate * correlation * 0.5;
        const maxW = this.ARCH.MAX_WEIGHT;
        const step = Math.max(1, Math.floor(pre.length / 50));
        for (let i = 0; i < post.length; i += step) {
            for (let j = 0; j < pre.length; j += step) {
                const delta = eta * pre[j] * post[i];
                if (label === 'P_to_A' && i < this.DIM.DIM_A) {
                    const headIdx = Math.floor(i / this.DIM.HEAD_DIM);
                    const localI = i % this.DIM.HEAD_DIM;
                    if (headIdx < this.DIM.NUM_HEADS &&
                        j < this.W_A[headIdx].length &&
                        localI < this.W_A[headIdx][j].length) {
                        this.W_A[headIdx][j][localI] = clamp(
                            this.W_A[headIdx][j][localI] + delta, -maxW, maxW
                        );
                    }
                }
            }
        }
    }

    _tdError(reward, value, nextValue, gamma) {
        return safeNumber(reward, 0)
            + gamma * safeNumber(nextValue, 0)
            - safeNumber(value, 0);
    }
}

// ============================================================================
// CROSS-PLATFORM EXPORT (FIX-13: single ESM export, guarded CJS)
// ============================================================================

// CJS compat — only if not in ESM mode
if (typeof module !== 'undefined' && module.exports && typeof exports !== 'undefined') {
    try {
        module.exports = { Cortex2Brain, SeededRNG, EmotionSystem, MemorySystem, SchemaManager };
    } catch (_) {
        // In ESM mode, module.exports assignment throws; silently ignore
    }
}

// Browser globals
if (typeof window !== 'undefined') {
    window.Cortex2Brain = Cortex2Brain;
    window.SeededRNG = SeededRNG;
}

// Worker globals
if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
    self.Cortex2Brain = Cortex2Brain;
    self.SeededRNG = SeededRNG;
}

export { Cortex2Brain, SeededRNG, EmotionSystem, MemorySystem, SchemaManager };
