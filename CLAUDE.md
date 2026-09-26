# Project: Explainable Auditory Attention Decoding

## Research goal
Develop a stimulus-conditioned graph-attention model for
EEG-based auditory attention decoding.

## Inputs
- One common multichannel EEG window
- Candidate Speech A
- Candidate Speech B

## Architecture
1. Extract temporal EEG features.
2. Extract frequency-domain EEG features.
3. Combine them into one feature vector per EEG electrode.
4. Use a physical electrode adjacency graph as the graph prior.
5. Encode Speech A and Speech B using one shared speech encoder.
6. Use the same candidate-conditioned Graph Attention Network
   for both candidates.
7. Speech should condition the graph-attention mechanism.
8. Produce one score for Candidate A and one score for Candidate B.
9. Compare the scores to predict the attended speaker.

## Interpretability
The model must expose:
- graph attention weights
- channel importance
- edge importance
- frequency-band contributions

Do not assume attention weights are faithful explanations.
Faithfulness will later be tested using ablation and perturbation.

## Development rules
- Use PyTorch.
- Keep modules separate and testable.
- Add tensor shape comments.
- Add unit tests.
- Do not change the architecture without discussing it first.
- Do not add extra Transformers, attention blocks, or losses automatically.