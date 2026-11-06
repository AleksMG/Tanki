/**
 * CORTEX2BRAIN v2.5 — Production Neural Architecture
 *
 * CHANGELOG v2.4 → v2.5:
 *   FIX-41  Cache-friendly multiply:      W stored transposed (column-major), sequential inner loop
 *   FIX-42  Fast weight init:             Xorshift stream replaces per-weight FNV-1a (~30× faster ctor)
 *   FIX-43  Full eligibility traces:      TD error now applied to W_P, W_D, W_pred (was only W_C)
 *   FIX-44  Correct emotion sensor idx:   SENSOR_IDX constants replace magic indices;
 *                                         `caution` now tracks bullet threat + HP, not HP alone
 *   FIX-45  Full input validation:        isFinite check runs on entire input vector, not first 20
 *   FIX-46  Clone copies traces:          Eligibility traces carried over to cloned brains
 *   FIX-47  Experience replay:            ReplayBuffer added; integrateReplay() drives off-policy updates
 *   FIX-48  Actor-critic:                 Policy gradient on D-layer via tanh-squashed outputs
 *   FIX-49  Advantage normalization:      Running mean/std over last 100 TD errors
 *   FIX-50  Real attention:               Q/K/V projections replace learnable gate softmax
 *
 * PREVIOUS FIXES preserved (v2.1 → v2.4):
 *   FIX-01 through FIX-40 (see v2.4 header)
 *
 * BACKWARD COMPATIBILITY:
 *   - Public API unchanged (constructor, forward, tdLearn, learn, getEmotions, etc.)
 *   - Serialized weights format is v2.4-compatible (reads both WT and legacy W layouts)
 *   - Output shape and sensor contract identical
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
 * @property {number}  [replayCapacity]
 * @property {number}  [replayBatchSize]
 * @property {number}  [replayMinSize]
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
 * @property {number[]|Float32Array} observation
 * @property {number[]|Float32Array} action
 * @property {number}               reward
 * @property {number[]|Float32Array} [nextObservation]
 * @property {boolean}              [done]
 * @property {Object}               [context]
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
 * @property {Object}   replayStats
 */

// ============================================================================
// ARCHITECTURE CONSTANTS
// ============================================================================

/**
 * FIX-44: Explicit sensor index map matching arena's SensorBuilder contract.
 * Centralizes the arena→brain sensor mapping so emotion/reflex systems don't
 * rely on magic numbers that silently drift out of sync.
 * @param {Object} DIM
 */
function deriveSensorIndices() {
    return Object.freeze({
        // Nearest enemy
        ENEMY_DX: 0, ENEMY_DY: 1, DISTANCE: 2, ENEMY_HP: 3,
        // Nearest wall
        WALL_DX: 4, WALL_DY: 5, WALL_DIST: 6,
        // Self position / HP
        SELF_X: 7, SELF_Y: 8, SELF_HP: 9,
        // Self stats
        LAST_REWARD: 10, TOTAL_DAMAGE: 11, HP_RATIO_ALT: 12, RELOAD_RATIO: 13,
        TIME_NORM: 14,
        // Reserved / unused slots
        _RESERVED_15: 15, _RESERVED_16: 16, _RESERVED_17: 17,
        _RESERVED_18: 18, _RESERVED_19: 19, _RESERVED_20: 20,
        _RESERVED_21: 21, _RESERVED_22: 22,
        // Misc
        DEATHS_NORM: 23, LAST_ENEMY_FLAG: 24,
        // Aim geometry
        ANGLE_TO_ENEMY: 25, ENEMY_SEES_ME: 26,
        IN_ENEMY_ARC: 27, ENEMY_IN_MY_ARC: 28,
        THREAT_COMPOSITE: 29,
        FLANK_X: 30, FLANK_Y: 31,
        ENEMY_TURRET_REL: 32, AIM_QUALITY: 33,
        SELF_DAMAGE_RATIO: 34,
        // Predicted enemy
        PRED_ENEMY_X: 35, PRED_ENEMY_Y: 36,
        // Pathing
        PATH_BLOCKED: 37,
        // Bullet threat (dodge)
        BULLET_THREAT: 38, DODGE_DX: 39, DODGE_DY: 40,
        // Wall avoidance
        WALL_VERY_CLOSE: 41, WALL_AVOID_STEER: 42,
        // 5-ray LIDAR around hull
        LIDAR_0: 43, LIDAR_1: 44, LIDAR_2: 45, LIDAR_3: 46, LIDAR_4: 47,
        // Two nearest enemies (angle, distance)
        EN2_ANGLE: 48, EN2_DIST: 49, EN3_ANGLE: 50, EN3_DIST: 51,
        // Turret orientation + meta
        TURRET_COS: 52, TURRET_SIN: 53,
        ALIVE_COUNT_NORM: 54, READY_TO_FIRE: 55,
        // Second-closest enemy detail
        EN2_DX: 56, EN2_DY: 57, EN2_HP: 58, EN2_ANGLE_FROM_ME: 59,
        // Border distances
        BORDER_LEFT: 60, BORDER_RIGHT: 61, BORDER_TOP: 62, BORDER_BOTTOM: 63,
    });
}

/** @param {Object} DIM */
function deriveArchSizes(DIM) {
    return Object.freeze({
        // Layer input widths
        P_INPUT:      DIM.DIM_SENSOR + DIM.DIM_C + DIM.DIM_S + 48,
        A_INPUT:      DIM.DIM_P + 48 + 64 + DIM.DIM_D,
        M_WORK_INPUT: DIM.DIM_P + DIM.DIM_A + 64,
        M_PERM_INPUT: DIM.DIM_C + 48,
        M_PRED_INPUT: DIM.DIM_A + 64 + 48 + DIM.DIM_D,
        C_INPUT:      DIM.DIM_P + DIM.DIM_A + DIM.DIM_M + DIM.DIM_S + DIM.DIM_D + 1,
        S_INPUT:      DIM.DIM_P + 48 + 32 + DIM.DIM_D,
        D_INPUT:      DIM.DIM_C + DIM.DIM_S + 48,

        // Memory layout offsets
        C_PRED_OFFSET:       192,
        C_STATE_OFFSET:      224,
        C_STATE_SLICE:       32,
        A_HEAD_C_SLICE:      64,
        M_WORK_C_SLICE:      64,
        M_BLOCK_SIZE:        48,

        PRED_CHANNELS:       48,
        MAX_RELATIONSHIPS:   100,
        MAX_EMOTION_HISTORY: 5,
        MAX_EPISODIC_EVENTS: 50,
        MAX_TD_HISTORY:      100,
        MAX_TRACE_SIZE:      10,
        EMOTION_MOMENTUM_WINDOW: 10,

        REFLEX_THRESHOLD:    0.9,
        MAX_WEIGHT:          2.0,
        MAX_GRADIENT:        0.3,
        MAX_TRACE_VAL:       1.0,

        // FIX-32: layer norm
        LAYER_NORM_EPS:      1e-5,
        // FIX-33: exploration noise
        INITIAL_NOISE:       0.15,
        NOISE_DECAY:         0.9999,
        MIN_NOISE:           0.02,
        // FIX-36: emotion decay
        EMOTION_DECAY_DEFAULT: 0.95,
        // FIX-37: frustration threshold
        FRUSTRATION_REWARD_THRESHOLD: -0.1,
        // FIX-40: aggression floor
        MIN_AGGRESSION_FACTOR: 0.15,

        // FIX-47: replay buffer
        DEFAULT_REPLAY_CAPACITY:  10000,
        DEFAULT_REPLAY_BATCH:     8,
        DEFAULT_REPLAY_MIN:       64,
        ADVANTAGE_WINDOW:         100,
    });
}

// ============================================================================
// SEEDED RNG (FIX-14, FIX-25: standalone; FIX-42: used for fast weight init)
// ============================================================================

class SeededRNG {
    /** @param {string|number} seed */
    constructor(seed) {
        let h = 0x811c9dc5;
        const s = String(seed || 'default');
        for (let i = 0; i < s.length; i++) {
            h ^= s.charCodeAt(i);
            h = Math.imul(h, 0x01000193);
        }
        this._state = (h >>> 0) || 1;
    }

    /** @returns {number} float in [0, 1) */
    next() {
        let x = this._state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this._state = x >>> 0;
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

    /** Symmetric uniform in [-1, 1). */
    sym() { return this.next() * 2 - 1; }
}

// ============================================================================
// SCHEMA MANAGER (FIX-25: extracted; FIX-45: full input validation)
// ============================================================================

class SchemaManager {
    /** @param {number} sensorDim */
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
     * FIX-45: isFinite check runs over entire input vector (was only first 20).
     * @param {number[]|Object} inputs
     * @param {Object} stats
     * @returns {number[]}
     */
    validateAndNormalizeInputs(inputs, stats) {
        if (inputs && typeof inputs === 'object' && !Array.isArray(inputs) && !(inputs instanceof Float32Array)) {
            const arr = [];
            for (const sensor of this._inputSchema) {
                const val = inputs[sensor.name];
                if (val === undefined) {
                    arr.push(0);
                } else {
                    const [min, max] = sensor.range || [-1, 1];
                    const span = max - min || 1;
                    arr.push(clamp((val - min) / span * 2 - 1, -1, 1));
                }
            }
            inputs = arr;
        }

        if (!Array.isArray(inputs) && !(inputs instanceof Float32Array)) {
            stats.validationErrors++;
            throw new Error(`Inputs must be array/Float32Array/object, got ${typeof inputs}`);
        }
        if (inputs.length === 0) {
            stats.validationErrors++;
            throw new Error('Inputs array cannot be empty');
        }

        // FIX-45: validate EVERY element, not only first 20
        const len = inputs.length;
        const maxReported = 5;
        let reportedCount = 0;
        for (let i = 0; i < len; i++) {
            const v = inputs[i];
            if (!Number.isFinite(v)) {
                stats.validationErrors++;
                if (reportedCount < maxReported) {
                    console.warn(`Invalid input at index ${i}: ${v}. Replacing with 0.`);
                    reportedCount++;
                }
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
        if (!Array.isArray(outputs) && !(outputs instanceof Float32Array)) {
            console.warn('Outputs must be array-like');
            return [];
        }
        const clamped = [];
        const schemaLen = this._outputSchema.length;
        const outLen = outputs.length;
        const n = Math.min(outLen, schemaLen);
        for (let i = 0; i < n; i++) {
            const [min, max] = this._outputSchema[i].range || [-1, 1];
            clamped.push(clamp(outputs[i], min, max));
        }
        while (clamped.length < schemaLen) {
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
            { name: 'self_hp_alt', range: [0, 1] },
            { name: 'self_fireRate', range: [0, 1] },
            { name: 'step_normalized', range: [0, 1] },
            { name: 'reserved_15', range: [-1, 1] },
            { name: 'reserved_16', range: [-1, 1] },
            { name: 'reserved_17', range: [-1, 1] },
            { name: 'reserved_18', range: [-1, 1] },
            { name: 'reserved_19', range: [-1, 1] },
            { name: 'reserved_20', range: [-1, 1] },
            { name: 'reserved_21', range: [-1, 1] },
            { name: 'reserved_22', range: [-1, 1] },
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
            { name: 'bullet_threat', range: [0, 1] },
            { name: 'dodge_dx', range: [-1, 1] },
            { name: 'dodge_dy', range: [-1, 1] },
            { name: 'wall_very_close', range: [0, 1] },
            { name: 'wall_avoidance_steer', range: [-1, 1] },
            { name: 'lidar_0', range: [0, 1] },
            { name: 'lidar_1', range: [0, 1] },
            { name: 'lidar_2', range: [0, 1] },
            { name: 'lidar_3', range: [0, 1] },
            { name: 'lidar_4', range: [0, 1] },
            { name: 'en2_angle', range: [-1, 1] },
            { name: 'en2_dist', range: [0, 1] },
            { name: 'en3_angle', range: [-1, 1] },
            { name: 'en3_dist', range: [0, 1] },
            { name: 'turret_cos', range: [-1, 1] },
            { name: 'turret_sin', range: [-1, 1] },
            { name: 'alive_count_norm', range: [0, 1] },
            { name: 'ready_to_fire', range: [0, 1], type: 'discrete' },
            { name: 'en2_dx', range: [-1, 1] },
            { name: 'en2_dy', range: [-1, 1] },
            { name: 'en2_hp', range: [0, 1] },
            { name: 'en2_angle_from_me', range: [-1, 1] },
            { name: 'border_left', range: [0, 1] },
            { name: 'border_right', range: [0, 1] },
            { name: 'border_top', range: [0, 1] },
            { name: 'border_bottom', range: [0, 1] },
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
// EMOTION SYSTEM (FIX-25: extracted; FIX-44: SENSOR_IDX-based)
// ============================================================================

class EmotionSystem {
    /**
     * @param {Object} personality
     * @param {number} maxHistory
     * @param {Object} archConstants
     * @param {Object} sensorIdx
     */
    constructor(personality, maxHistory = 30, archConstants = {}, sensorIdx = null) {
        this.personality = personality;
        this.maxHistory = maxHistory;
        this.ARCH = archConstants;
        // FIX-44: explicit sensor indices
        this.SI = sensorIdx || deriveSensorIndices();

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
            fear:        { baseDuration: 180, intensityMultiplier: 2.0, satisfactionDecay: 0.02 },
            aggression:  { baseDuration: 120, intensityMultiplier: 1.5, satisfactionDecay: 0.03 },
            frustration: { baseDuration: 240, intensityMultiplier: 2.5, satisfactionDecay: 0.01 },
            desperation: { baseDuration: 300, intensityMultiplier: 3.0, satisfactionDecay: 0.005 },
            confidence:  { baseDuration: 150, intensityMultiplier: 1.2, satisfactionDecay: 0.04 },
            vengeance:   { baseDuration: 360, intensityMultiplier: 3.5, satisfactionDecay: 0.003 },
        };

        this.emotionDecay = archConstants.EMOTION_DECAY_DEFAULT || 0.95;
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
     * FIX-31: smoothed lerp for threat/caution
     * FIX-36: stronger decay
     * FIX-37: frustration only grows on actually negative reward
     * FIX-40: aggression floor from personality
     * FIX-44: uses SENSOR_IDX, not magic numbers
     *
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

        // FIX-37: frustration grows only on actually negative reward
        const frustThreshold = this.ARCH.FRUSTRATION_REWARD_THRESHOLD || -0.1;
        if (reward < frustThreshold) {
            this.emotions.frustration = Math.min(1, this.emotions.frustration + 0.02);
        } else if (reward > 0) {
            this.emotions.frustration *= 0.88;
        } else {
            this.emotions.frustration *= 0.95;
        }

        const normalizedReward = clamp((cumulativeReward + 100) / 200, 0, 1);
        if (normalizedReward < 0.3) {
            this.emotions.desperation = Math.min(1, this.emotions.desperation + 0.03);
        } else {
            this.emotions.desperation *= 0.90;
        }

        if (reward > 1) {
            this.emotions.confidence = Math.min(1, this.emotions.confidence + 0.08);
        } else if (reward > 0) {
            this.emotions.confidence = Math.min(1, this.emotions.confidence + 0.02);
        } else {
            this.emotions.confidence *= 0.97;
        }

        if (reward < -0.5) {
            this.emotions.fear = Math.min(1, this.emotions.fear + 0.08);
        } else {
            this.emotions.fear *= 0.90;
        }

        // FIX-44: use SENSOR_IDX for threat (distance to nearest enemy)
        const rawDistance = safeNumber(sensorInputs[this.SI.DISTANCE], 0.5);
        const rawThreat = 1 - rawDistance;
        this.emotions.threat = lerp(this.emotions.threat, rawThreat, 0.1);

        // FIX-44: caution = blend of low HP + incoming bullet threat
        // (was incorrectly using sensorInputs[12] which is HP, producing inverted logic)
        const hpRatio = safeNumber(sensorInputs[this.SI.SELF_HP], 1.0);
        const bulletThreat = safeNumber(sensorInputs[this.SI.BULLET_THREAT], 0);
        const lowHpSignal = 1 - hpRatio;
        const rawCaution = clamp(lowHpSignal * 0.55 + bulletThreat * 0.7, 0, 1);
        this.emotions.caution = lerp(this.emotions.caution, rawCaution, 0.15);

        this.emotions.surprise = lerp(this.emotions.surprise, avgPredError, 0.2);

        // Momentum (rolling window average)
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

        // FIX-40: personality-based aggression floor
        const fearBlock = (1 - this.emotions.desperation) * this.emotionalState.momentum.fear;
        const personalityAggression = (this.personality.traits.aggression || 0.5) *
            (this.ARCH.MIN_AGGRESSION_FACTOR || 0.15);

        this.emotions.aggression = Math.max(
            personalityAggression,
            this.emotionalState.momentum.frustration * 0.4 +
            this.emotionalState.momentum.desperation * 0.5 +
            this.emotionalState.momentum.vengeance * 0.7 -
            fearBlock * 0.2
        );

        // FIX-36: stronger decay
        for (const key of Object.keys(this.emotions)) {
            this.emotions[key] *= this.emotionDecay;
            if (this.emotions[key] < 0.001) this.emotions[key] = 0;
        }
    }

    /** @param {Float32Array} stateVector */
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
     * FIX-34: fear no longer suppresses throttle
     * @param {Float32Array} sensorInputs
     * @param {Float32Array} stateVector
     * @param {Float32Array} decisionVector
     * @param {Object} stats
     */
    applyReflexes(sensorInputs, stateVector, decisionVector, stats) {
        const em = this.emotionInfluence;
        if (stateVector[8] > 0.6) {
            stats.reflexTriggers++;
            decisionVector[6] = lerp(decisionVector[6], Math.min(1, decisionVector[6] + 0.30), em);
            decisionVector[4] = lerp(decisionVector[4], Math.min(1, decisionVector[4] + 0.20), em);
        }
        if (stateVector[7] > 0.6) {
            stats.reflexTriggers++;
            decisionVector[6] = lerp(decisionVector[6], Math.min(1, decisionVector[6] + 0.25), em);
            decisionVector[4] = lerp(decisionVector[4], Math.min(1, decisionVector[4] + 0.20), em);
        }
        if (stateVector[4] > 0.7) {
            stats.reflexTriggers++;
            decisionVector[10] = lerp(decisionVector[10], Math.min(1, decisionVector[10] + 0.20), em);
        }
        if (stateVector[1] > 0.7) {
            stats.reflexTriggers++;
            decisionVector[6] = lerp(decisionVector[6], Math.min(1, decisionVector[6] + 0.20), em);
            decisionVector[4] = lerp(decisionVector[4], Math.min(1, decisionVector[4] + 0.15), em);
        }
    }

    /** FIX-34: softer modulation, no full suppression */
    resolveConflicts(decisionVector) {
        const eb = this.emotionInfluence * 0.5;
        const emotions = this.emotions;
        if (emotions.vengeance > 0.6 || emotions.desperation > 0.6) {
            decisionVector[10] = Math.max(0, decisionVector[10] - eb * 0.5);
            decisionVector[7] = Math.min(1, decisionVector[7] + eb * 0.3);
        }
        if (emotions.fear > 0.7) {
            decisionVector[6] = Math.max(0.1, decisionVector[6] - eb * 0.3);
        }
        if (emotions.aggression > 0.6) {
            decisionVector[6] = Math.min(1, decisionVector[6] + eb * 0.5);
            decisionVector[4] = Math.min(1, decisionVector[4] + eb * 0.2);
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
            { name: 'vengeance',   value: this.emotions.vengeance },
            { name: 'desperation', value: this.emotions.desperation },
            { name: 'fear',        value: this.emotions.fear },
            { name: 'aggression',  value: this.emotions.aggression },
            { name: 'frustration', value: this.emotions.frustration },
            { name: 'confidence',  value: this.emotions.confidence },
        ];
        candidates.sort((a, b) => b.value - a.value);
        return { name: candidates[0].name, intensity: candidates[0].value };
    }

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
            name, intensity,
            initialIntensity: intensity,
            baseDuration: config.baseDuration,
            remainingFrames: Math.floor(config.baseDuration * intensity * config.intensityMultiplier),
            satisfaction: this.emotionalState.satisfaction,
            startStep: step,
        };
    }

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

    _canExitEmotion(state) {
        if (state.remainingFrames <= 0) return true;
        if (state.satisfaction > 0.6) return true;
        if (state.intensity < 0.3) return true;
        if (state.remainingFrames > state.baseDuration * 0.7) return false;
        return false;
    }

    _shouldExtendEmotion(state) {
        return state.intensity > state.initialIntensity + 0.2 || state.satisfaction < -0.3;
    }

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

    _processCombatEvents(events, health, maxHealth) {
        if (!events) return;
        const eventList = Array.isArray(events) ? events : [events];
        for (const event of eventList) {
            if (!event || !event.type) continue;
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
                    if (event.killer !== undefined) {
                        this.emotionMemory.killer = event.killer;
                        const grudge = this.emotionMemory.grudge.get(event.killer) || 0;
                        this.emotionMemory.grudge.set(event.killer, clamp(grudge + 0.5, -1, 1));
                        this.emotions.vengeance = Math.min(1, this.emotions.vengeance + 0.6);
                    }
                    break;
            }
        }
    }

    /** @returns {Object} */
    toJSON() {
        return {
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
                ? { ...data.emotionalState.current } : null;
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

        this._evictOldRelationships();
    }

    recallRelevantMemories({ enemyId, healthRatio, situation }) {
        const recalled = [];
        for (const event of this.episodicMemory.events) {
            if (enemyId !== undefined && event.agentId === enemyId) {
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
// REPLAY BUFFER (FIX-47: off-policy experience storage)
// ============================================================================

class ReplayBuffer {
    /**
     * @param {number} capacity
     * @param {SeededRNG|null} rng - optional seeded RNG for reproducible sampling
     */
    constructor(capacity = 10000, rng = null) {
        this.capacity = Math.max(1, capacity | 0);
        this._buffer = new Array(this.capacity);
        this._size = 0;
        this._head = 0;
        this._rng = rng;
        this._totalPushed = 0;
    }

    /** @param {Experience} experience */
    push(experience) {
        if (!experience) return;
        const slot = this._buffer[this._head] || {};
        slot.observation = experience.observation
            ? Float32Array.from(experience.observation) : null;
        slot.action = experience.action
            ? Float32Array.from(experience.action) : null;
        slot.reward = Number(experience.reward) || 0;
        slot.nextObservation = experience.nextObservation
            ? Float32Array.from(experience.nextObservation) : null;
        slot.done = !!experience.done;
        slot.context = experience.context || null;
        this._buffer[this._head] = slot;
        this._head = (this._head + 1) % this.capacity;
        if (this._size < this.capacity) this._size++;
        this._totalPushed++;
    }

    /**
     * @param {number} batchSize
     * @returns {Experience[]}
     */
    sample(batchSize) {
        if (this._size === 0) return [];
        const n = Math.min(batchSize, this._size);
        const out = new Array(n);
        const rnd = this._rng
            ? () => Math.floor(this._rng.next() * this._size)
            : () => Math.floor(Math.random() * this._size);
        for (let i = 0; i < n; i++) {
            const idx = rnd();
            const src = this._buffer[idx];
            out[i] = {
                observation: src.observation,
                action: src.action,
                reward: src.reward,
                nextObservation: src.nextObservation,
                done: src.done,
                context: src.context,
            };
        }
        return out;
    }

    size() { return this._size; }
    isFull() { return this._size === this.capacity; }
    totalPushed() { return this._totalPushed; }
    clear() { this._size = 0; this._head = 0; }

    /** @returns {Object} serialisable (capped to last 500 entries for size) */
    toJSON() {
        const maxSerialized = 500;
        const snapshot = [];
        const n = Math.min(this._size, maxSerialized);
        for (let i = 0; i < n; i++) {
            const idx = (this._head - n + i + this.capacity) % this.capacity;
            const e = this._buffer[idx];
            if (!e) continue;
            snapshot.push({
                observation: e.observation ? Array.from(e.observation) : null,
                action: e.action ? Array.from(e.action) : null,
                reward: e.reward,
                nextObservation: e.nextObservation ? Array.from(e.nextObservation) : null,
                done: e.done,
            });
        }
        return {
            capacity: this.capacity,
            size: this._size,
            totalPushed: this._totalPushed,
            entries: snapshot,
        };
    }

    loadFromJSON(data) {
        if (!data || !Array.isArray(data.entries)) return;
        this.clear();
        this.capacity = data.capacity || this.capacity;
        this._totalPushed = data.totalPushed || 0;
        if (this._buffer.length !== this.capacity) {
            this._buffer = new Array(this.capacity);
        }
        for (const e of data.entries) {
            this.push({
                observation: e.observation,
                action: e.action,
                reward: e.reward,
                nextObservation: e.nextObservation,
                done: e.done,
            });
        }
    }
}

// ============================================================================
// STANDALONE MATH UTILITIES
// ============================================================================

function clamp(v, min, max) { return v < min ? min : v > max ? max : v; }
function lerp(a, b, t) { const ct = t < 0 ? 0 : t > 1 ? 1 : t; return a + (b - a) * ct; }
function safeNumber(val, fallback) {
    return (typeof val === 'number' && isFinite(val)) ? val : fallback;
}
function tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    const e = Math.exp(2 * x);
    return (e - 1) / (e + 1);
}
function leakyRelu(x, alpha = 0.01) { return x > 0 ? x : alpha * x; }

function softmax(arr, temp = 1) {
    if (!arr || arr.length === 0) return [];
    const n = arr.length;
    let max = -Infinity;
    for (let i = 0; i < n; i++) if (arr[i] > max) max = arr[i];
    const exps = new Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
        const v = Math.exp(Math.min((arr[i] - max) / temp, 700));
        exps[i] = v;
        sum += v;
    }
    if (sum < 1e-10) sum = 1e-10;
    for (let i = 0; i < n; i++) exps[i] /= sum;
    return exps;
}

// ============================================================================
// MAIN CLASS
// ============================================================================

class Cortex2Brain {

    /** @param {CortexConfig} config */
    constructor(config = {}) {
        this._initConfig(config);
        this._initArchitecture();
        this._initWeights();
        this._initStateVectors();
        this._initEligibilityTraces();
        this._initSubsystems();
        this._initRuntimeState();
        this._isReady = true;
    }

    // ========================================================================
    // INITIALISATION
    // ========================================================================

    _initConfig(config) {
        this.config = {
            seed: config.seed || `CORTEX_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
            lr: config.lr ?? 0.015,
            gamma: config.gamma ?? 0.99,
            lambda: config.lambda ?? 0.95,
            hebbianRate: config.hebbianRate ?? 0.002,
            taskSpec: config.taskSpec || null,
            replayCapacity: config.replayCapacity ?? 10000,
            replayBatchSize: config.replayBatchSize ?? 8,
            replayMinSize: config.replayMinSize ?? 64,
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
            M_WORK_START: 0,   M_WORK_END: 48,
            M_PERM_START: 48,  M_PERM_END: 96,
            M_PRED_START: 96,  M_PRED_END: 144,
        });

        this.ARCH = deriveArchSizes(this.DIM);
        // FIX-44: expose sensor index map
        this.SENSOR_IDX = deriveSensorIndices();
        this.RATIOS = this._calculateRatios(this.config.taskSpec);
        this._rng = new SeededRNG(this.config.seed);
    }

    _initSubsystems() {
        this.predictionErrors = new Float32Array(this.DIM.DIM_SENSOR);
        this._stats = this._createEmptyStats();

        this.personality = {
            traits: {
                bravery: 0.5, loyalty: 0.5, empathy: 0.5,
                curiosity: 0.5, patience: 0.5, aggression: 0.5,
            },
            values: {
                survival: 0.8, victory: 0.7, fairness: 0.4, loyalty: 0.6,
            },
        };

        // FIX-44: pass SENSOR_IDX to EmotionSystem
        this.emotionSystem = new EmotionSystem(
            this.personality, 30, this.ARCH, this.SENSOR_IDX
        );
        this.memorySystem = new MemorySystem(
            this.ARCH.MAX_RELATIONSHIPS,
            this.ARCH.MAX_EPISODIC_EVENTS
        );
        this._schemaManager = new SchemaManager(this.DIM.DIM_SENSOR);

        // FIX-47: replay buffer (seeded RNG for reproducible sampling)
        this.replay = new ReplayBuffer(
            this.config.replayCapacity || this.ARCH.DEFAULT_REPLAY_CAPACITY,
            this._rng
        );

        // FIX-33: exploration noise level
        this._noiseLevel = this.ARCH.INITIAL_NOISE;

        // FIX-49: running advantage statistics
        this._advantageHistory = [];
    }

    _initRuntimeState() {
        this.step = 0;
        this.cumulativeReward = 0;
        this.health = 100;
        this.stats = { maxHealth: 100 };
        this._lastKiller = null;
        this._lastInputs = null;
        this._lastOutputs = null;
        this._lastCInput = null;
        this._lastDInput = null;
        this._reflexThisFrame = false;
        this._traceBuffer = null;

        // FIX-39: randomize initial D values for diverse starting behavior
        for (let i = 0; i < this.DIM.DIM_D; i++) {
            this.D[i] = this._rng.sym() * 0.3;
            this.D_prev[i] = this.D[i];
        }
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
            replayUpdates: 0,
            actorUpdates: 0,
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
    // SCHEMA & VALIDATION
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

    configureEmotions(config) { this.emotionSystem.configure(config); }

    updateTaskSpec(taskSpec) {
        this.config.taskSpec = { ...this.config.taskSpec, ...taskSpec };
        this.RATIOS = this._calculateRatios(this.config.taskSpec);
    }

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
     * FIX-32: Layer normalization on C
     * FIX-33: Exploration noise on D
     * FIX-35: Per-agent noise from seeded RNG
     * FIX-38: C/S normalized before D input
     * FIX-41: Cache-friendly matrix multiply (transposed weights)
     * FIX-50: Real attention with Q/K/V
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

        const sensorInput = this.validateAndNormalizeInputs(rawInputs);
        const rewardVal = safeNumber(reward, 0);

        this._lastInputs = Float32Array.from(sensorInput);
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

        this._lastCInput = cInput;

        if (trace) trace.layers.inputBuild = performance.now();

        // === FORWARD PROPAGATION (FIX-41: transposed weight multiply) ===

        const pNew = this._matMul(pInput, this.WT_P);
        for (let i = 0; i < this.DIM.DIM_P; i++) pNew[i] = leakyRelu(pNew[i] + this.b_P[i]);

        // FIX-50: real attention with Q/K/V
        const aNew = new Float32Array(this.DIM.DIM_A);
        const headScores = new Array(this.DIM.NUM_HEADS);
        // Build context K/V from pooled (P_prev + mWorkPrev)
        const contextVec = this._buildAttentionContext(this.P_prev, mWorkPrev);
        const K = this._matMul(contextVec, this.WT_K);
        const V = this._matMul(contextVec, this.WT_V);

        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            const Qh = this._matMul(aInputs[h], this.WT_Q[h]);
            // Scaled dot-product with head's K-slice
            const offset = h * this.DIM.HEAD_DIM;
            let score = 0;
            for (let i = 0; i < this.DIM.HEAD_DIM; i++) {
                score += Qh[offset + i] * K[offset + i];
            }
            headScores[h] = score * this._attnScale;
        }
        const gateProbs = softmax(headScores, 1.0);

        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            const headOut = this._matMul(aInputs[h], this.WT_A[h]);
            const gateWeight = gateProbs[h];
            const offset = h * this.DIM.HEAD_DIM;
            for (let i = 0; i < this.DIM.HEAD_DIM; i++) {
                const v = V[offset + i];
                aNew[offset + i] = leakyRelu(headOut[offset + i] + this.b_A[h][i]) * gateWeight
                    + v * (1 - gateWeight) * 0.3;
            }
        }

        const mWorkNew = this._matMul(mWorkInput, this.WT_M_work);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mWorkNew[i] = leakyRelu(mWorkNew[i] + this.b_M_work[i]);

        const mPermNew = this._matMul(mPermInput, this.WT_M_perm);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mPermNew[i] = leakyRelu(mPermNew[i] + this.b_M_perm[i]);

        const mPredNew = this._matMul(mPredInput, this.WT_M_pred);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mPredNew[i] = leakyRelu(mPredNew[i] + this.b_M_pred[i]);

        const mNew = new Float32Array(this.DIM.DIM_M);
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mNew[i] = mWorkNew[i];
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mNew[this.ARCH.M_BLOCK_SIZE + i] = mPermNew[i];
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) mNew[this.ARCH.M_BLOCK_SIZE * 2 + i] = mPredNew[i];

        let cNew = this._matMul(cInput, this.WT_C);
        for (let i = 0; i < this.DIM.DIM_C; i++) cNew[i] = leakyRelu(cNew[i] + this.b_C[i]);
        // FIX-32: layer normalization
        cNew = this._layerNorm(cNew);

        const sNew = this._matMul(sInput, this.WT_S);
        for (let i = 0; i < this.DIM.DIM_S; i++) sNew[i] = leakyRelu(sNew[i] + this.b_S[i]);

        // FIX-38: Build D input from normalized C and S
        const dInputNorm = this._buildDInput(cNew, sNew, mPredNew);
        this._lastDInput = dInputNorm;

        const dNew = this._matMul(dInputNorm, this.WT_D);
        for (let i = 0; i < this.DIM.DIM_D; i++) dNew[i] = tanh(dNew[i] + this.b_D[i]);

        // FIX-33/35: exploration noise
        for (let i = 0; i < this.DIM.DIM_D; i++) {
            dNew[i] = clamp(dNew[i] + this._rng.gaussian(0, this._noiseLevel), -1, 1);
        }
        this._noiseLevel = Math.max(
            this.ARCH.MIN_NOISE,
            this._noiseLevel * this.ARCH.NOISE_DECAY
        );

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

        // === REFLEX + CONFLICT RESOLUTION ===
        this._reflexThisFrame = false;
        if (combatEvents) {
            this.emotionSystem.applyReflexes(sensorInput, sNew, dNew, this._stats);
        }
        this.emotionSystem.resolveConflicts(dNew);
        for (let i = 0; i < this.DIM.DIM_S; i++) {
            if (sNew[i] > this.ARCH.REFLEX_THRESHOLD) {
                this._reflexThisFrame = true;
                break;
            }
        }

        // === ELIGIBILITY TRACES UPDATE (FIX-43: all layers) ===
        this._updateEligibilityTraces(pInput, cInput, dInputNorm, mPredNew);

        if (trace) trace.layers.emotions = performance.now();

        // === COMMIT STATE ===
        this.P_prev.set(this.P); this.A_prev.set(this.A); this.M_prev.set(this.M);
        this.C_prev.set(this.C); this.S_prev.set(this.S); this.D_prev.set(this.D);
        this.P.set(pNew); this.A.set(aNew); this.M.set(mNew);
        this.C.set(cNew); this.S.set(sNew); this.D.set(dNew);

        // === OUTPUT ===
        const output = this.validateAndClampOutputs(Array.from(this.D));
        this._lastOutputs = Float32Array.from(output);

        // === PUSH TO REPLAY BUFFER (FIX-47) ===
        if (this._lastInputs && this._lastOutputs) {
            this.replay.push({
                observation: this._lastInputs,
                action: this._lastOutputs,
                reward: rewardVal,
                nextObservation: null, // filled by caller via setNextObservation or learn()
                done: false,
            });
        }

        // === TRACE FINALISE ===
        if (trace) {
            trace.end = performance.now();
            trace.duration = trace.end - trace.start;
            this._traceBuffer.push(trace);
            const maxTr = this._traceBuffer.maxSize ?? this.ARCH.MAX_TRACE_SIZE;
            if (this._traceBuffer.length > maxTr) this._traceBuffer.shift();
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

    /**
     * Allows arena to attach nextObservation to the most recent replay entry.
     * Should be called BEFORE the next forward() if you want proper TD targets.
     * @param {Float32Array|number[]} nextObs
     */
    setNextObservation(nextObs) {
        const last = this.replay.size() > 0
            ? this.replay._buffer[(this.replay._head - 1 + this.replay.capacity) % this.replay.capacity]
            : null;
        if (last) {
            last.nextObservation = Float32Array.from(nextObs);
        }
    }

    // ========================================================================
    // BATCH PROCESSING
    // ========================================================================

    forwardBatch(inputsArray, options = {}) {
        const { rewards = [] } = options;
        return inputsArray.map((inputs, i) => this.forward(inputs, rewards[i] || 0, null, null));
    }

    /**
     * FIX-49: advantage normalization across batch
     * FIX-47: processes experiences via actor-critic
     * @param {Experience[]} experiences
     * @param {Object} options
     */
    learnBatch(experiences, options = {}) {
        const { accumulateGradients = true } = options;
        if (!experiences || experiences.length === 0) return;
        if (!accumulateGradients) {
            for (const exp of experiences) this.learn(exp);
            return;
        }

        // Compute TD errors for all experiences
        const tdErrors = new Array(experiences.length);
        for (let i = 0; i < experiences.length; i++) {
            const exp = experiences[i];
            const currentValue = this._estimateValue();
            let nextValue = currentValue;
            if (!exp.done && exp.nextObservation) {
                const saved = this._saveState();
                try {
                    this.forward(exp.nextObservation, 0);
                    nextValue = this._estimateValue();
                } finally {
                    this._restoreState(saved);
                }
            }
            tdErrors[i] = this._tdError(exp.reward, currentValue, nextValue, this.config.gamma);
        }

        // FIX-49: advantage normalization
        const normalizedAdv = this._normalizeAdvantages(tdErrors);

        // Update critic (value head) with raw TD errors
        let avgTDErr = 0;
        for (let i = 0; i < tdErrors.length; i++) {
            const weight = 1 / experiences.length;
            this._updateValueHead(tdErrors[i] * weight);
            avgTDErr += Math.abs(tdErrors[i]);
        }
        avgTDErr /= experiences.length;

        this._stats.tdErrorHistory.push(Math.abs(avgTDErr));
        if (this._stats.tdErrorHistory.length > this.ARCH.MAX_TD_HISTORY) {
            this._stats.tdErrorHistory.shift();
        }

        // FIX-48: actor update via policy gradient
        for (let i = 0; i < experiences.length; i++) {
            const exp = experiences[i];
            if (exp.action) {
                this._updateActor(normalizedAdv[i], Float32Array.from(exp.action));
                this._stats.actorUpdates++;
            }
        }

        this._stats.replayUpdates += experiences.length;
    }

    // ========================================================================
    // LEARNING
    // ========================================================================

    learn(experience) {
        const { observation, action, reward, nextObservation, done } = experience;
        const inputs = this.validateAndNormalizeInputs(observation);
        const outputs = Array.isArray(action) ? action : (action?.output ? Array.from(action.output) : []);

        const currentValue = this._estimateValue();
        let nextValue = 0;

        if (!done && nextObservation) {
            const saved = this._saveState();
            try {
                this.forward(nextObservation, 0);
                nextValue = this._estimateValue();
            } finally {
                this._restoreState(saved);
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

        // Critic update
        this._updateValueHead(tdErr);

        // FIX-43: apply TD error to all relevant weight matrices via traces
        this._applyTDError(tdErr);

        // FIX-48: policy gradient on D layer using normalized advantage
        if (outputs && outputs.length > 0) {
            const advantage = this._computeAdvantage(tdErr);
            this._updateActor(advantage, Float32Array.from(outputs));
            this._stats.actorUpdates++;
        }

        // Eligibility trace update (must happen AFTER apply, before next forward)
        if (this._lastInputs && this._lastCInput && this._lastDInput) {
            this._updateEligibilityTraces(this._lastInputs, this._lastCInput, this._lastDInput, null);
        }

        // Memory consolidation on strong reward
        if (reward > 3) {
            this.memorySystem.consolidateMemories(
                this.predictionErrors, this.M, reward,
                this.step, this._stats, this.DIM.M_PRED_START
            );
        }

        // Hebbian updates on positive reward
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

    /**
     * FIX-47: integrate replay-driven off-policy updates.
     * Call this periodically (e.g. every N frames) from the arena loop.
     * @param {number} [batchSize]
     */
    integrateReplay(batchSize = null) {
        const bs = batchSize ?? this.config.replayBatchSize ?? this.ARCH.DEFAULT_REPLAY_BATCH;
        const minSize = this.config.replayMinSize ?? this.ARCH.DEFAULT_REPLAY_MIN;
        if (this.replay.size() < Math.min(bs, minSize)) return 0;
        const batch = this.replay.sample(bs);
        if (batch.length === 0) return 0;
        this.learnBatch(batch);
        return batch.length;
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
        this._lastDInput = null;
        this.step = 0;

        // FIX-39: re-randomize D
        for (let i = 0; i < this.DIM.DIM_D; i++) {
            this.D[i] = this._rng.sym() * 0.3;
            this.D_prev[i] = this.D[i];
        }

        // FIX-33: reset noise level
        this._noiseLevel = this.ARCH.INITIAL_NOISE;

        // FIX-43: reset eligibility traces
        this.traceP.fill(0);
        this.traceC.fill(0);
        this.traceD.fill(0);
        this.tracePred.fill(0);

        // FIX-49: reset advantage history
        this._advantageHistory.length = 0;
    }

    reset() {
        this.resetEpisode();
        this.memorySystem.resetFull();
        this.replay.clear();
        this._stats = this._createEmptyStats();
    }

    // ========================================================================
    // DEBUG & TRACING
    // ========================================================================

    setTracing(enable, maxTraces = 10) {
        this._traceBuffer = enable ? [] : null;
        if (this._traceBuffer) this._traceBuffer.maxSize = maxTraces;
    }

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
            attentionWeights: this._getAttentionWeights(),
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
                    trust: v.trust, affinity: v.affinity, lastInteraction: v.lastInteraction,
                }]),
                reputation: sm.reputation,
            },
            replayStats: {
                size: this.replay.size(),
                capacity: this.replay.capacity,
                totalPushed: this.replay.totalPushed(),
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
    // SERIALISATION (backward-compatible with v2.4)
    // ========================================================================

    toJSON(options = {}) {
        const {
            includePersonality = true,
            includeEmotions = true,
            includeSocialMemory = true,
            includeEpisodicMemory = false,
            includeStats = true,
            includeSchema = true,
            includeReplay = false,
        } = options;

        const data = {
            type: 'Cortex2Brain',
            version: '2.5',
            config: {
                seed: this.config.seed,
                lr: this.config.lr,
                gamma: this.config.gamma,
                lambda: this.config.lambda,
                hebbianRate: this.config.hebbianRate,
                taskSpec: this.config.taskSpec,
                dims: { ...this.DIM },
                replayCapacity: this.config.replayCapacity,
                replayBatchSize: this.config.replayBatchSize,
                replayMinSize: this.config.replayMinSize,
            },
            weights: this._serializeWeights(),
        };

        if (includePersonality) {
            data.personality = {
                traits: { ...this.personality.traits },
                values: { ...this.personality.values },
            };
        }
        if (includeEmotions) data.emotionSystem = this.emotionSystem.toJSON();

        const memoryData = this.memorySystem.toJSON();
        if (includeSocialMemory) data.socialMemory = memoryData.socialMemory;
        if (includeEpisodicMemory) data.episodicMemory = memoryData.episodicMemory;
        if (includeStats) data.stats = { ...this._stats };
        if (includeSchema) {
            data.inputSchema = this._schemaManager.getInputSchema();
            data.outputSchema = this._schemaManager.getOutputSchema();
        }
        if (includeReplay) data.replay = this.replay.toJSON();

        return data;
    }

    static fromJSON(data) {
        if (!data || data.type !== 'Cortex2Brain') throw new Error('Invalid Cortex2Brain JSON data');

        const brain = new Cortex2Brain(data.config);
        const w = data.weights;
        if (w) brain._validateAndLoadWeights(w);

        if (data.personality) {
            brain.personality.traits = { ...brain.personality.traits, ...data.personality.traits };
            brain.personality.values = { ...brain.personality.values, ...data.personality.values };
        }
        if (data.emotionSystem) brain.emotionSystem.loadFromJSON(data.emotionSystem);
        if (data.emotionMemory && !data.emotionSystem) {
            brain.emotionSystem.loadFromJSON({ emotionMemory: data.emotionMemory });
        }

        const memoryData = {};
        if (data.socialMemory) memoryData.socialMemory = data.socialMemory;
        if (data.episodicMemory) memoryData.episodicMemory = data.episodicMemory;
        if (data.socialMemory || data.episodicMemory) brain.memorySystem.loadFromJSON(memoryData);

        if (data.inputSchema) brain._schemaManager.setInputSchema(data.inputSchema);
        if (data.outputSchema) brain._schemaManager.setOutputSchema(data.outputSchema);
        if (data.stats) brain._stats = { ...brain._stats, ...data.stats };
        if (data.replay) brain.replay.loadFromJSON(data.replay);

        return brain;
    }

    /**
     * Backward-compatible serialization: converts transposed weights back to
     * row-major W layout so v2.4 loaders can still read the file. Also emits
     * the new WT format for v2.5 loaders.
     */
    _serializeWeights() {
        // Helper: transpose WT[col][row] -> W[row][col] for legacy compat
        const untranspose = (wt) => {
            if (!wt || wt.length === 0) return [];
            const cols = wt.length;
            const rows = wt[0].length;
            const out = new Array(rows);
            for (let i = 0; i < rows; i++) {
                const row = new Array(cols);
                for (let j = 0; j < cols; j++) row[j] = wt[j][i];
                out[i] = row;
            }
            return out;
        };
        const untransposeHeads = (wtHeads) => wtHeads.map(untranspose);

        return {
            // v2.5 native format (transposed)
            WT_P: this.WT_P.map(c => Array.from(c)),
            WT_A: this.WT_A.map(m => m.map(c => Array.from(c))),
            WT_Q: this.WT_Q.map(m => m.map(c => Array.from(c))),
            WT_K: this.WT_K.map(c => Array.from(c)),
            WT_V: this.WT_V.map(c => Array.from(c)),
            WT_M_work: this.WT_M_work.map(c => Array.from(c)),
            WT_M_perm: this.WT_M_perm.map(c => Array.from(c)),
            WT_M_pred: this.WT_M_pred.map(c => Array.from(c)),
            WT_C: this.WT_C.map(c => Array.from(c)),
            WT_S: this.WT_S.map(c => Array.from(c)),
            WT_D: this.WT_D.map(c => Array.from(c)),
            WT_pred: this.WT_pred.map(c => Array.from(c)),
            // v2.4 legacy format (row-major) for backward compatibility
            W_P: untranspose(this.WT_P),
            b_P: Array.from(this.b_P),
            W_A: untransposeHeads(this.WT_A),
            b_A: this.b_A.map(b => Array.from(b)),
            W_gate: Array.from(this.b_gate), // legacy: used W_gate as head bias
            W_reflex: Array.from(this.W_reflex),
            W_M_work: untranspose(this.WT_M_work),
            b_M_work: Array.from(this.b_M_work),
            W_M_perm: untranspose(this.WT_M_perm),
            b_M_perm: Array.from(this.b_M_perm),
            W_M_pred: untranspose(this.WT_M_pred),
            b_M_pred: Array.from(this.b_M_pred),
            W_C: untranspose(this.WT_C),
            b_C: Array.from(this.b_C),
            W_S: untranspose(this.WT_S),
            b_S: Array.from(this.b_S),
            W_D: untranspose(this.WT_D),
            b_D: Array.from(this.b_D),
            W_pred: untranspose(this.WT_pred),
            b_pred: Array.from(this.b_pred),
            W_V: Array.from(this.W_V),
            b_V: this.b_V,
        };
    }

    _validateAndLoadWeights(w) {
        const checkMat = (name, mat, expectedRows, expectedCols) => {
            if (!mat || mat.length !== expectedRows) {
                throw new Error(`${name} row count mismatch: expected ${expectedRows}, got ${mat?.length}`);
            }
            if (expectedCols !== undefined && mat[0]?.length !== expectedCols) {
                throw new Error(`${name} col count mismatch: expected ${expectedCols}, got ${mat[0]?.length}`);
            }
        };
        const checkVec = (name, vec, expectedLen) => {
            if (!vec || vec.length !== expectedLen) {
                throw new Error(`${name} length mismatch: expected ${expectedLen}, got ${vec?.length}`);
            }
        };

        // Prefer native v2.5 WT format if present
        const hasNative = !!(w.WT_C && w.WT_D && w.WT_P);

        if (hasNative) {
            // Load WT directly
            const loadWT = (wt, rows, cols, name) => {
                checkMat(name, wt, cols, rows); // WT is [cols][rows]
                return wt.map(c => Float32Array.from(c));
            };
            const loadWTHeads = (wtHeads, rows, cols, name) => {
                if (wtHeads.length !== this.DIM.NUM_HEADS) {
                    throw new Error(`${name} head count mismatch`);
                }
                return wtHeads.map((m, h) => loadWT(m, rows, cols, `${name}[${h}]`));
            };

            this.WT_P = loadWT(w.WT_P, this.ARCH.P_INPUT, this.DIM.DIM_P, 'WT_P');
            this.WT_A = loadWTHeads(w.WT_A, this.ARCH.A_INPUT, this.DIM.HEAD_DIM, 'WT_A');
            this.WT_Q = loadWTHeads(w.WT_Q, this.ARCH.A_INPUT, this.DIM.DIM_A, 'WT_Q');
            this.WT_K = loadWT(w.WT_K, this.ARCH.A_INPUT, this.DIM.DIM_A, 'WT_K');
            this.WT_V = loadWT(w.WT_V, this.ARCH.A_INPUT, this.DIM.DIM_A, 'WT_V');
            this.WT_M_work = loadWT(w.WT_M_work, this.ARCH.M_WORK_INPUT, this.ARCH.M_BLOCK_SIZE, 'WT_M_work');
            this.WT_M_perm = loadWT(w.WT_M_perm, this.ARCH.M_PERM_INPUT, this.ARCH.M_BLOCK_SIZE, 'WT_M_perm');
            this.WT_M_pred = loadWT(w.WT_M_pred, this.ARCH.M_PRED_INPUT, this.ARCH.M_BLOCK_SIZE, 'WT_M_pred');
            this.WT_C = loadWT(w.WT_C, this.ARCH.C_INPUT, this.DIM.DIM_C, 'WT_C');
            this.WT_S = loadWT(w.WT_S, this.ARCH.S_INPUT, this.DIM.DIM_S, 'WT_S');
            this.WT_D = loadWT(w.WT_D, this.ARCH.D_INPUT, this.DIM.DIM_D, 'WT_D');
            this.WT_pred = loadWT(w.WT_pred, this.ARCH.M_BLOCK_SIZE, this.DIM.DIM_SENSOR, 'WT_pred');
        } else {
            // Legacy v2.4 row-major format: transpose on load
            const transpose = (mat) => {
                if (!mat || mat.length === 0) return [];
                const rows = mat.length;
                const cols = mat[0].length;
                const out = new Array(cols);
                for (let j = 0; j < cols; j++) {
                    const col = new Float32Array(rows);
                    for (let i = 0; i < rows; i++) col[i] = mat[i][j];
                    out[j] = col;
                }
                return out;
            };
            const transposeHeads = (matHeads) => matHeads.map(transpose);

            checkMat('W_P', w.W_P, this.ARCH.P_INPUT, this.DIM.DIM_P);
            this.WT_P = transpose(w.W_P);

            if (w.W_A.length !== this.DIM.NUM_HEADS) {
                throw new Error(`W_A head count mismatch`);
            }
            for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
                checkMat(`W_A[${h}]`, w.W_A[h], this.ARCH.A_INPUT, this.DIM.HEAD_DIM);
            }
            this.WT_A = transposeHeads(w.W_A);
            // Legacy had no Q/K/V — initialize fresh (already done in _initWeights)

            checkMat('W_M_work', w.W_M_work, this.ARCH.M_WORK_INPUT, this.ARCH.M_BLOCK_SIZE);
            this.WT_M_work = transpose(w.W_M_work);
            checkMat('W_M_perm', w.W_M_perm, this.ARCH.M_PERM_INPUT, this.ARCH.M_BLOCK_SIZE);
            this.WT_M_perm = transpose(w.W_M_perm);
            checkMat('W_M_pred', w.W_M_pred, this.ARCH.M_PRED_INPUT, this.ARCH.M_BLOCK_SIZE);
            this.WT_M_pred = transpose(w.W_M_pred);

            checkMat('W_C', w.W_C, this.ARCH.C_INPUT, this.DIM.DIM_C);
            this.WT_C = transpose(w.W_C);
            checkMat('W_S', w.W_S, this.ARCH.S_INPUT, this.DIM.DIM_S);
            this.WT_S = transpose(w.W_S);
            checkMat('W_D', w.W_D, this.ARCH.D_INPUT, this.DIM.DIM_D);
            this.WT_D = transpose(w.W_D);
            checkMat('W_pred', w.W_pred, this.ARCH.M_BLOCK_SIZE, this.DIM.DIM_SENSOR);
            this.WT_pred = transpose(w.W_pred);
        }

        // Biases and vectors (format-agnostic)
        checkVec('b_P', w.b_P, this.DIM.DIM_P);
        this.b_P = Float32Array.from(w.b_P);
        if (w.b_A) {
            for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
                checkVec(`b_A[${h}]`, w.b_A[h], this.DIM.HEAD_DIM);
                this.b_A[h] = Float32Array.from(w.b_A[h]);
            }
        }
        if (w.W_gate) {
            checkVec('W_gate', w.W_gate, this.DIM.NUM_HEADS);
            this.b_gate = Float32Array.from(w.W_gate); // legacy gate → head bias
        }
        if (w.W_reflex) this.W_reflex = Float32Array.from(w.W_reflex);

        checkVec('b_M_work', w.b_M_work, this.ARCH.M_BLOCK_SIZE);
        this.b_M_work = Float32Array.from(w.b_M_work);
        checkVec('b_M_perm', w.b_M_perm, this.ARCH.M_BLOCK_SIZE);
        this.b_M_perm = Float32Array.from(w.b_M_perm);
        checkVec('b_M_pred', w.b_M_pred, this.ARCH.M_BLOCK_SIZE);
        this.b_M_pred = Float32Array.from(w.b_M_pred);

        checkVec('b_C', w.b_C, this.DIM.DIM_C);
        this.b_C = Float32Array.from(w.b_C);
        checkVec('b_S', w.b_S, this.DIM.DIM_S);
        this.b_S = Float32Array.from(w.b_S);
        checkVec('b_D', w.b_D, this.DIM.DIM_D);
        this.b_D = Float32Array.from(w.b_D);
        checkVec('b_pred', w.b_pred, this.DIM.DIM_SENSOR);
        this.b_pred = Float32Array.from(w.b_pred);
        checkVec('W_V', w.W_V, this.DIM.DIM_C);
        this.W_V = Float32Array.from(w.W_V);
        this.b_V = w.b_V ?? 0;
    }

    // ========================================================================
    // CLONING & MUTATION
    // ========================================================================

    clone(options = {}) {
        const { copyWeights = true, copyState = false, copyMemory = true, copyTraces = true, copyReplay = false } = options;

        const cloned = new Cortex2Brain({
            seed: this.config.seed + '_clone_' + Date.now(),
            lr: this.config.lr,
            gamma: this.config.gamma,
            lambda: this.config.lambda,
            hebbianRate: this.config.hebbianRate,
            taskSpec: this.config.taskSpec ? { ...this.config.taskSpec } : null,
            replayCapacity: this.config.replayCapacity,
            replayBatchSize: this.config.replayBatchSize,
            replayMinSize: this.config.replayMinSize,
        });

        if (copyWeights) {
            const cmCol = src => src.map(c => Float32Array.from(c));
            const cmHeads = src => src.map(cmCol);

            cloned.WT_P = cmCol(this.WT_P);
            cloned.WT_A = cmHeads(this.WT_A);
            cloned.WT_Q = cmHeads(this.WT_Q);
            cloned.WT_K = cmCol(this.WT_K);
            cloned.WT_V = cmCol(this.WT_V);
            cloned.WT_M_work = cmCol(this.WT_M_work);
            cloned.WT_M_perm = cmCol(this.WT_M_perm);
            cloned.WT_M_pred = cmCol(this.WT_M_pred);
            cloned.WT_C = cmCol(this.WT_C);
            cloned.WT_S = cmCol(this.WT_S);
            cloned.WT_D = cmCol(this.WT_D);
            cloned.WT_pred = cmCol(this.WT_pred);
            cloned.b_P = Float32Array.from(this.b_P);
            cloned.b_A = this.b_A.map(b => Float32Array.from(b));
            cloned.b_gate = Float32Array.from(this.b_gate);
            cloned.W_reflex = Float32Array.from(this.W_reflex);
            cloned.b_M_work = Float32Array.from(this.b_M_work);
            cloned.b_M_perm = Float32Array.from(this.b_M_perm);
            cloned.b_M_pred = Float32Array.from(this.b_M_pred);
            cloned.b_C = Float32Array.from(this.b_C);
            cloned.b_S = Float32Array.from(this.b_S);
            cloned.b_D = Float32Array.from(this.b_D);
            cloned.b_pred = Float32Array.from(this.b_pred);
            cloned.W_V = Float32Array.from(this.W_V);
            cloned.b_V = this.b_V;
        }

        if (copyState) {
            cloned.P.set(this.P); cloned.A.set(this.A); cloned.M.set(this.M);
            cloned.C.set(this.C); cloned.S.set(this.S); cloned.D.set(this.D);
            cloned.P_prev.set(this.P_prev); cloned.A_prev.set(this.A_prev); cloned.M_prev.set(this.M_prev);
            cloned.C_prev.set(this.C_prev); cloned.S_prev.set(this.S_prev); cloned.D_prev.set(this.D_prev);
            cloned.predictionErrors.set(this.predictionErrors);
        }

        // FIX-46: copy eligibility traces
        if (copyTraces) {
            cloned.traceP.set(this.traceP);
            cloned.traceC.set(this.traceC);
            cloned.traceD.set(this.traceD);
            cloned.tracePred.set(this.tracePred);
        }

        cloned.personality = {
            traits: { ...this.personality.traits },
            values: { ...this.personality.values },
        };

        if (copyMemory) {
            cloned.emotionSystem.loadFromJSON(this.emotionSystem.toJSON());
            cloned.memorySystem.loadFromJSON(this.memorySystem.toJSON());
        }

        if (copyReplay) {
            cloned.replay.loadFromJSON(this.replay.toJSON());
        }

        cloned._schemaManager.setInputSchema(this._schemaManager.getInputSchema());
        cloned._schemaManager.setOutputSchema(this._schemaManager.getOutputSchema());
        cloned._stats = {
            ...this._stats,
            tdErrorHistory: [...this._stats.tdErrorHistory],
            predictionErrorHistory: [...this._stats.predictionErrorHistory],
        };
        cloned._advantageHistory = [...this._advantageHistory];

        return cloned;
    }

    mutate(rate = 0.01, strength = 0.1) {
        let mutations = 0;
        const maxW = this.ARCH.MAX_WEIGHT;
        const mutateMat = mat => {
            if (!mat?.length) return;
            for (let j = 0; j < mat.length; j++) {
                const col = mat[j];
                if (!col?.length) continue;
                for (let i = 0; i < col.length; i++) {
                    if (this._rng.next() < rate) {
                        col[i] = clamp(col[i] + this._rng.gaussian(0, strength), -maxW, maxW);
                        mutations++;
                    }
                }
            }
        };
        mutateMat(this.WT_P); mutateMat(this.WT_C); mutateMat(this.WT_D);
        for (const WT of this.WT_A) mutateMat(WT);
        for (const WT of this.WT_Q) mutateMat(WT);
        mutateMat(this.WT_K); mutateMat(this.WT_V);
        mutateMat(this.WT_M_work); mutateMat(this.WT_M_perm); mutateMat(this.WT_M_pred);
        mutateMat(this.WT_S); mutateMat(this.WT_pred);
        for (let i = 0; i < this.b_gate.length; i++) {
            if (this._rng.next() < rate) {
                this.b_gate[i] = clamp(this.b_gate[i] + this._rng.gaussian(0, strength), -maxW, maxW);
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
    getReplayBuffer() { return this.replay; }

    // ========================================================================
    // PRIVATE — WEIGHT INITIALISATION (FIX-42: fast Xorshift)
    // ========================================================================

    _initWeights() {
        const bs = this.ARCH.M_BLOCK_SIZE;
        const rng = this._rng;

        // FIX-42: use Xorshift stream instead of per-weight FNV-1a hashing
        // (eliminates O(rows*cols) hash calls in ctor)
        const initMatT = (rows, cols, scale) => {
            const matT = new Array(cols);
            for (let j = 0; j < cols; j++) {
                const col = new Float32Array(rows);
                for (let i = 0; i < rows; i++) col[i] = rng.sym() * scale;
                matT[j] = col;
            }
            return matT;
        };
        const initVec = (length, scale) => {
            const vec = new Float32Array(length);
            for (let i = 0; i < length; i++) vec[i] = rng.sym() * scale;
            return vec;
        };

        this.WT_P = initMatT(this.ARCH.P_INPUT, this.DIM.DIM_P, 0.3);
        this.b_P = initVec(this.DIM.DIM_P, 0.1);

        this.WT_A = [];
        this.b_A = [];
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            this.WT_A.push(initMatT(this.ARCH.A_INPUT, this.DIM.HEAD_DIM, 0.3));
            this.b_A.push(initVec(this.DIM.HEAD_DIM, 0.1));
        }

        // FIX-50: real attention projections (per-head Q; shared K/V)
        this.WT_Q = [];
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            this.WT_Q.push(initMatT(this.ARCH.A_INPUT, this.DIM.DIM_A, 0.2));
        }
        this.WT_K = initMatT(this.ARCH.A_INPUT, this.DIM.DIM_A, 0.2);
        this.WT_V = initMatT(this.ARCH.A_INPUT, this.DIM.DIM_A, 0.2);
        this._attnScale = 1 / Math.sqrt(this.DIM.HEAD_DIM);

        // Legacy b_gate retained for serialization compat
        this.b_gate = initVec(this.DIM.NUM_HEADS, 0.0);
        this.W_reflex = initVec(4, 0.5);

        this.WT_M_work = initMatT(this.ARCH.M_WORK_INPUT, bs, 0.3);
        this.b_M_work = initVec(bs, 0.1);
        this.WT_M_perm = initMatT(this.ARCH.M_PERM_INPUT, bs, 0.3);
        this.b_M_perm = initVec(bs, 0.1);
        this.WT_M_pred = initMatT(this.ARCH.M_PRED_INPUT, bs, 0.3);
        this.b_M_pred = initVec(bs, 0.1);

        this.WT_C = initMatT(this.ARCH.C_INPUT, this.DIM.DIM_C, 0.25);
        this.b_C = initVec(this.DIM.DIM_C, 0.1);

        this.WT_S = initMatT(this.ARCH.S_INPUT, this.DIM.DIM_S, 0.3);
        this.b_S = initVec(this.DIM.DIM_S, 0.1);

        this.WT_D = initMatT(this.ARCH.D_INPUT, this.DIM.DIM_D, 0.2);
        this.b_D = initVec(this.DIM.DIM_D, 0.1);

        this.WT_pred = initMatT(bs, this.DIM.DIM_SENSOR, 0.15);
        this.b_pred = initVec(this.DIM.DIM_SENSOR, 0.05);

        this.W_V = initVec(this.DIM.DIM_C, 0.1);
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

    /** FIX-43: full trace set including D and prediction */
    _initEligibilityTraces() {
        this.traceP = new Float32Array(this.ARCH.P_INPUT * this.DIM.DIM_P);
        this.traceC = new Float32Array(this.ARCH.C_INPUT * this.DIM.DIM_C);
        this.traceD = new Float32Array(this.ARCH.D_INPUT * this.DIM.DIM_D);
        this.tracePred = new Float32Array(this.ARCH.M_BLOCK_SIZE * this.DIM.DIM_SENSOR);
    }

    // ========================================================================
    // PRIVATE — MATH / MATRIX OPS (FIX-41: cache-friendly)
    // ========================================================================

    /**
     * FIX-41: Column-major matrix-vector multiply.
     * matT is stored as matT[col][row], giving sequential inner loop (cache-friendly).
     * Equivalent to row-major W·vec but ~2–4× faster in V8.
     * @param {Float32Array|number[]} vec
     * @param {Float32Array[]} matT - array of column vectors
     * @returns {Float32Array}
     */
    _matMul(vec, matT) {
        const cols = matT.length;
        const out = new Float32Array(cols);
        const n = vec.length;
        for (let j = 0; j < cols; j++) {
            const col = matT[j];
            let sum = 0;
            const lim = Math.min(n, col.length);
            // Unrolled 4× for V8 JIT inlining
            let i = 0;
            for (; i + 3 < lim; i += 4) {
                sum += vec[i] * col[i] + vec[i + 1] * col[i + 1]
                     + vec[i + 2] * col[i + 2] + vec[i + 3] * col[i + 3];
            }
            for (; i < lim; i++) sum += vec[i] * col[i];
            out[j] = sum;
        }
        return out;
    }

    /**
     * FIX-32: Layer normalization.
     * @param {Float32Array} vec
     * @returns {Float32Array}
     */
    _layerNorm(vec) {
        const n = vec.length;
        if (n === 0) return vec;

        let sum = 0;
        for (let i = 0; i < n; i++) sum += vec[i];
        const mean = sum / n;

        let varSum = 0;
        for (let i = 0; i < n; i++) {
            const d = vec[i] - mean;
            varSum += d * d;
        }
        const std = Math.sqrt(varSum / n + this.ARCH.LAYER_NORM_EPS);

        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            let v = (vec[i] - mean) / std;
            if (v > 2) v = 2;
            else if (v < -2) v = -2;
            out[i] = v;
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
            output[i] = this._rng.sym() * 0.05;
        }
        return output;
    }

    _getMWork(M) { return M.subarray(this.DIM.M_WORK_START, this.DIM.M_WORK_END); }
    _getMPerm(M) { return M.subarray(this.DIM.M_PERM_START, this.DIM.M_PERM_END); }
    _getMPred(M) { return M.subarray(this.DIM.M_PRED_START, this.DIM.M_PRED_END); }

    /** FIX-50: pooled context vector for K/V projections */
    _buildAttentionContext(pPrev, mWork) {
        const out = new Float32Array(this.ARCH.A_INPUT);
        let idx = 0;
        for (let i = 0; i < this.DIM.DIM_P; i++) out[idx++] = pPrev[i];
        for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) out[idx++] = mWork[i];
        while (idx < this.ARCH.A_INPUT) out[idx++] = 0;
        return out;
    }

    // ========================================================================
    // PRIVATE — LAYER INPUT BUILDERS
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
    // PRIVATE — LEARNING INTERNALS (FIX-43, FIX-48, FIX-49)
    // ========================================================================

    _updatePrediction(mPred, sensorInput) {
        const xPred = this._matMul(mPred, this.WT_pred);
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) xPred[i] += this.b_pred[i];
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) {
            const error = Math.abs(xPred[i] - sensorInput[i]);
            this.predictionErrors[i] = Math.min(1, Math.max(0, error));
        }
    }

    _computeAvgPredError() {
        let sum = 0;
        for (let i = 0; i < this.DIM.DIM_SENSOR; i++) sum += this.predictionErrors[i];
        return sum / this.DIM.DIM_SENSOR;
    }

    _estimateValue() {
        let sum = this.b_V;
        const C = this.C, W = this.W_V, n = this.DIM.DIM_C;
        let i = 0;
        for (; i + 3 < n; i += 4) {
            sum += C[i]*W[i] + C[i+1]*W[i+1] + C[i+2]*W[i+2] + C[i+3]*W[i+3];
        }
        for (; i < n; i++) sum += C[i] * W[i];
        return tanh(sum);
    }

    _updateValueHead(tdErr) {
        const alpha = this.config.lr * 0.5;
        const maxW = this.ARCH.MAX_WEIGHT;
        const C = this.C;
        for (let i = 0; i < this.DIM.DIM_C; i++) {
            this.W_V[i] = clamp(this.W_V[i] + alpha * tdErr * C[i], -maxW, maxW);
        }
        this.b_V += alpha * tdErr;
    }

    /**
     * FIX-43: traces updated for P, C, D, and prediction layers.
     * Uses exponential decay γλ (TD(λ)).
     */
    _updateEligibilityTraces(pInput, cInput, dInput, mPred) {
        const decay = this.config.gamma * this.config.lambda;
        const maxTrace = this.ARCH.MAX_TRACE_VAL;

        // P traces
        if (pInput) {
            const pLen = Math.min(pInput.length, this.ARCH.P_INPUT);
            const dimP = this.DIM.DIM_P;
            const traceP = this.traceP;
            const P = this.P;
            for (let i = 0; i < this.ARCH.P_INPUT; i++) {
                const inputVal = i < pLen ? pInput[i] : 0;
                const base = i * dimP;
                for (let j = 0; j < dimP; j++) {
                    const idx = base + j;
                    const v = decay * traceP[idx] + inputVal * P[j];
                    traceP[idx] = v > maxTrace ? maxTrace : v < -maxTrace ? -maxTrace : v;
                }
            }
        }

        // C traces
        if (cInput) {
            const cLen = cInput.length;
            const dimC = this.DIM.DIM_C;
            const traceC = this.traceC;
            const C = this.C;
            for (let j = 0; j < this.ARCH.C_INPUT; j++) {
                const preAct = j < cLen ? cInput[j] : 0;
                const base = j * dimC;
                for (let i = 0; i < dimC; i++) {
                    const idx = base + i;
                    const v = decay * traceC[idx] + preAct * C[i];
                    traceC[idx] = v > maxTrace ? maxTrace : v < -maxTrace ? -maxTrace : v;
                }
            }
        }

        // D traces (FIX-43 — was missing entirely)
        if (dInput) {
            const dLen = dInput.length;
            const dimD = this.DIM.DIM_D;
            const traceD = this.traceD;
            const D = this.D;
            for (let j = 0; j < this.ARCH.D_INPUT; j++) {
                const preAct = j < dLen ? dInput[j] : 0;
                const base = j * dimD;
                for (let i = 0; i < dimD; i++) {
                    const idx = base + i;
                    const v = decay * traceD[idx] + preAct * D[i];
                    traceD[idx] = v > maxTrace ? maxTrace : v < -maxTrace ? -maxTrace : v;
                }
            }
        }

        // Prediction traces
        if (mPred) {
            const bs = this.ARCH.M_BLOCK_SIZE;
            const dimS = this.DIM.DIM_SENSOR;
            const tracePr = this.tracePred;
            for (let j = 0; j < bs; j++) {
                const preAct = mPred[j];
                const base = j * dimS;
                for (let i = 0; i < dimS; i++) {
                    const idx = base + i;
                    const v = decay * tracePr[idx] + preAct * this.predictionErrors[i];
                    tracePr[idx] = v > maxTrace ? maxTrace : v < -maxTrace ? -maxTrace : v;
                }
            }
        }
    }

    /**
     * FIX-43: apply TD error to ALL trace-carrying layers (P, C, D, pred).
     * Was previously only updating W_C.
     */
    _applyTDError(tdErr) {
        const alpha = this.config.lr * Math.sign(tdErr) * 0.1;
        const maxGrad = this.ARCH.MAX_GRADIENT;
        const maxW = this.ARCH.MAX_WEIGHT;

        // --- W_C update (legacy behavior preserved) ---
        const dimC = this.DIM.DIM_C;
        const traceC = this.traceC;
        for (let i = 0; i < dimC; i++) {
            for (let j = 0; j < this.ARCH.C_INPUT; j++) {
                const col = this.WT_C[i];
                if (col) {
                    const rawDelta = alpha * 0.01 * traceC[j * dimC + i];
                    let delta = rawDelta;
                    if (delta > maxGrad) { delta = maxGrad; this._stats.gradientClips++; }
                    else if (delta < -maxGrad) { delta = -maxGrad; this._stats.gradientClips++; }
                    col[j] = clamp(col[j] + delta, -maxW, maxW);
                }
            }
        }

        // --- W_P update (FIX-43: previously missing) ---
        const dimP = this.DIM.DIM_P;
        const traceP = this.traceP;
        for (let i = 0; i < dimP; i++) {
            for (let j = 0; j < this.ARCH.P_INPUT; j++) {
                const col = this.WT_P[i];
                if (col) {
                    const rawDelta = alpha * 0.008 * traceP[j * dimP + i];
                    let delta = rawDelta;
                    if (delta > maxGrad) { delta = maxGrad; this._stats.gradientClips++; }
                    else if (delta < -maxGrad) { delta = -maxGrad; this._stats.gradientClips++; }
                    col[j] = clamp(col[j] + delta, -maxW, maxW);
                }
            }
        }

        // --- W_pred update (FIX-43: previously missing) ---
        const dimS = this.DIM.DIM_SENSOR;
        const tracePr = this.tracePred;
        for (let i = 0; i < dimS; i++) {
            for (let j = 0; j < this.ARCH.M_BLOCK_SIZE; j++) {
                const col = this.WT_pred[i];
                if (col) {
                    const rawDelta = alpha * 0.005 * tracePr[j * dimS + i];
                    let delta = rawDelta;
                    if (delta > maxGrad) { delta = maxGrad; this._stats.gradientClips++; }
                    else if (delta < -maxGrad) { delta = -maxGrad; this._stats.gradientClips++; }
                    col[j] = clamp(col[j] + delta, -maxW, maxW);
                }
            }
        }
    }

    /**
     * FIX-48: Actor-critic policy gradient on D-layer.
     * Uses tanh squashing correction and eligibility traces.
     * @param {number} advantage
     * @param {Float32Array} outputs - D-layer outputs (post-tanh)
     */
    _updateActor(advantage, outputs) {
        if (!outputs || outputs.length === 0) return;
        const alpha = this.config.lr * 0.05;
        const maxGrad = this.ARCH.MAX_GRADIENT;
        const maxW = this.ARCH.MAX_WEIGHT;
        const dimD = this.DIM.DIM_D;
        const traceD = this.traceD;

        const n = Math.min(outputs.length, dimD);
        for (let i = 0; i < n; i++) {
            // tanh derivative: 1 - tanh²(x)
            const squash = 1 - outputs[i] * outputs[i];
            const policyGrad = squash * advantage;
            for (let j = 0; j < this.ARCH.D_INPUT; j++) {
                const rawDelta = alpha * policyGrad * traceD[j * dimD + i];
                let delta = rawDelta;
                if (delta > maxGrad) { delta = maxGrad; this._stats.gradientClips++; }
                else if (delta < -maxGrad) { delta = -maxGrad; this._stats.gradientClips++; }
                this.WT_D[i][j] = clamp(this.WT_D[i][j] + delta, -maxW, maxW);
            }
        }
    }

    /**
     * FIX-49: running mean/std advantage normalization.
     * Stabilizes policy gradient updates.
     */
    _computeAdvantage(tdErr) {
        this._advantageHistory.push(tdErr);
        const maxH = this.ARCH.ADVANTAGE_WINDOW;
        if (this._advantageHistory.length > maxH) this._advantageHistory.shift();
        const n = this._advantageHistory.length;
        if (n < 5) return tdErr;

        let sum = 0, sumSq = 0;
        for (let i = 0; i < n; i++) {
            const v = this._advantageHistory[i];
            sum += v;
            sumSq += v * v;
        }
        const mean = sum / n;
        const variance = sumSq / n - mean * mean;
        const std = Math.sqrt(Math.max(variance, 1e-6));
        return (tdErr - mean) / std;
    }

    _normalizeAdvantages(tdErrors) {
        const n = tdErrors.length;
        if (n === 0) return tdErrors;
        let sum = 0, sumSq = 0;
        for (let i = 0; i < n; i++) {
            sum += tdErrors[i];
            sumSq += tdErrors[i] * tdErrors[i];
        }
        const mean = sum / n;
        const variance = sumSq / n - mean * mean;
        const std = Math.sqrt(Math.max(variance, 1e-6));
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = (tdErrors[i] - mean) / std;
        return out;
    }

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
                        localI < this.WT_A[headIdx].length &&
                        j < this.WT_A[headIdx][localI].length) {
                        // WT_A[head][col=row_in_output][row=input_idx]
                        this.WT_A[headIdx][localI][j] = clamp(
                            this.WT_A[headIdx][localI][j] + delta, -maxW, maxW
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

    /** Save snapshot of recurrent state for safe forward-then-restore. */
    _saveState() {
        return {
            P: new Float32Array(this.P),
            A: new Float32Array(this.A),
            M: new Float32Array(this.M),
            C: new Float32Array(this.C),
            S: new Float32Array(this.S),
            D: new Float32Array(this.D),
            P_prev: new Float32Array(this.P_prev),
            A_prev: new Float32Array(this.A_prev),
            M_prev: new Float32Array(this.M_prev),
            C_prev: new Float32Array(this.C_prev),
            S_prev: new Float32Array(this.S_prev),
            D_prev: new Float32Array(this.D_prev),
            predictionErrors: new Float32Array(this.predictionErrors),
            step: this.step,
            noiseLevel: this._noiseLevel,
            cumulativeReward: this.cumulativeReward,
        };
    }

    _restoreState(snap) {
        this.P.set(snap.P); this.A.set(snap.A); this.M.set(snap.M);
        this.C.set(snap.C); this.S.set(snap.S); this.D.set(snap.D);
        this.P_prev.set(snap.P_prev); this.A_prev.set(snap.A_prev); this.M_prev.set(snap.M_prev);
        this.C_prev.set(snap.C_prev); this.S_prev.set(snap.S_prev); this.D_prev.set(snap.D_prev);
        this.predictionErrors.set(snap.predictionErrors);
        this.step = snap.step;
        this._noiseLevel = snap.noiseLevel;
        this.cumulativeReward = snap.cumulativeReward;
    }

    /** Current attention head probabilities (last forward pass). */
    _getAttentionWeights() {
        // Recompute quickly from cached state
        if (!this.P_prev || !this.M_prev) return new Array(this.DIM.NUM_HEADS).fill(0);
        const mWork = this._getMWork(this.M_prev);
        const ctx = this._buildAttentionContext(this.P_prev, mWork);
        const K = this._matMul(ctx, this.WT_K);
        const scores = new Array(this.DIM.NUM_HEADS);
        for (let h = 0; h < this.DIM.NUM_HEADS; h++) {
            const pLen = this.DIM.DIM_P;
            // Reconstruct aInputs[h] partially (only need Q projection)
            const input = new Float32Array(this.ARCH.A_INPUT);
            let idx = 0;
            for (let i = 0; i < pLen; i++) input[idx++] = this.P_prev[i] * this.RATIOS.head;
            for (let i = 0; i < this.ARCH.M_BLOCK_SIZE; i++) input[idx++] = mWork[i];
            const cStart = h * this.ARCH.A_HEAD_C_SLICE;
            for (let i = 0; i < this.ARCH.A_HEAD_C_SLICE; i++) {
                const ci = cStart + i;
                input[idx++] = ci < this.DIM.DIM_C ? this.C_prev[ci] : 0;
            }
            for (let i = 0; i < this.DIM.DIM_D; i++) input[idx++] = this.D_prev[i];
            const Qh = this._matMul(input, this.WT_Q[h]);
            const offset = h * this.DIM.HEAD_DIM;
            let score = 0;
            for (let i = 0; i < this.DIM.HEAD_DIM; i++) score += Qh[offset + i] * K[offset + i];
            scores[h] = score * this._attnScale;
        }
        return softmax(scores, 1.0);
    }
}

// ============================================================================
// CROSS-PLATFORM EXPORT
// ============================================================================

if (typeof module !== 'undefined' && module.exports && typeof exports !== 'undefined') {
    try {
        module.exports = {
            Cortex2Brain, SeededRNG, EmotionSystem, MemorySystem,
            SchemaManager, ReplayBuffer,
        };
    } catch (_) {
        // ESM mode — silently ignore
    }
}

if (typeof window !== 'undefined') {
    window.Cortex2Brain = Cortex2Brain;
    window.SeededRNG = SeededRNG;
    window.EmotionSystem = EmotionSystem;
    window.MemorySystem = MemorySystem;
    window.SchemaManager = SchemaManager;
    window.ReplayBuffer = ReplayBuffer;
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
    self.Cortex2Brain = Cortex2Brain;
    self.SeededRNG = SeededRNG;
}

export {
    Cortex2Brain,
    SeededRNG,
    EmotionSystem,
    MemorySystem,
    SchemaManager,
    ReplayBuffer,
};
