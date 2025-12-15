# Related Work: Emotion Regulation in AI

This document summarizes existing approaches to compare ARC against.

## 1. Existing Approaches

### 1.1 Affective Computing (Recognition-focused)
- **Picard (1997)**: Pioneered emotion recognition in AI
- **Focus**: External expression (facial, voice, text)
- **Gap**: Does NOT address internal regulation or stability

### 1.2 Emotion as Reinforcement Signal
- **Moerland et al. (2018)**: Emotion-driven RL exploration
- **Focus**: Emotions boost exploration/exploitation tradeoff
- **Gap**: No homeostatic control, no anti-rumination

### 1.3 Rational Emotional Patterns (REM)
- **OpenAI Forum (2024)**: Mathematical model of emotions as "attractors"
- **Focus**: Stabilize interaction dynamics
- **Gap**: No learning integration, no safety mechanisms

### 1.4 Responsible Reinforcement Learning (RRL)
- **ArXiv (2024)**: Multi-objective reward for behavioral health
- **Focus**: Balance engagement with user well-being
- **Gap**: External user emotions, not agent internal state

## 2. What Makes ARC Novel

| Feature | REM | RRL | Affective RL | **ARC** |
|---------|-----|-----|--------------|---------|
| Internal state regulation | Partial | ❌ | ❌ | ✅ |
| Anti-rumination (DMN) | ❌ | ❌ | ❌ | ✅ |
| Memory gating | ❌ | ❌ | ❌ | ✅ |
| Hierarchical/Meta-control | ❌ | ❌ | Partial | ✅ |
| Safety under adversarial | ❌ | ❌ | ❌ | ✅ |
| RL integration (L6) | ❌ | ✅ | ✅ | ✅ |

## 3. Key Differentiators

1. **PFC-Inspired Architecture**: ARC models prefrontal cortex regulation functions
2. **DMN Suppression**: Explicit mechanism against narrative-driven rumination
3. **Homeostatic Control**: Arousal/uncertainty thresholds with proportional response
4. **Validated Benchmark (ASSB)**: Reproducible metrics across perturbation scenarios
5. **RL Transfer Learning**: Demonstrated improvement in non-stationary environments

## 4. Fair Comparison Note

Most existing work focuses on:
- **Emotion recognition** (not regulation)
- **User emotions** (not agent internal state)
- **Single objective** (not multi-objective with safety)

ARC is the first framework to combine:
- Affective state dynamics (IIT, GWT, PP-inspired)
- Control theory (proportional + gain scheduling)
- RL integration (learning rate/exploration modulation)
- Safety adversarial testing (L5 scenarios)

## 5. References

- Picard, R.W. (1997). Affective Computing. MIT Press.
- Moerland, T.M. et al. (2018). Emotion in Reinforcement Learning Agents and Robots. Machine Learning, 107(2).
- Scherer, K.R. et al. (2010). Blueprint for Affective Computing. Oxford.
- Responsible RL (2024). arXiv preprint.
- REM model discussions: OpenAI Community Forum.
