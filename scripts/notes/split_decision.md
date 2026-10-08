Question: Which states will be used for training, validation, and hidden evaluation?
This is a provisional allocation for exploring geographic and temporal generalization. The EDA supports the data’s usability and shows differences in observed income-positive rates. It does not establish how difficult any state will be for a model.
Which states?
Training: CA, TX, and NY (2018). Validation: FL (2018). Hidden evaluation: MS, WV, NM, AR, LA, and MT (2018 and 2021).
Why these training states?
California, Texas, and New York provide substantial samples from different regions. This gives the model varied training data, though it does not guarantee broad or representative coverage.
Why Florida for validation?
Florida is outside the training states and has enough records for validation. Its shift difficulty is not yet known.
Why these hidden states?
They provide six geographic populations outside the training states for evaluating transfer. Their differing income-positive rates suggest marginal differences, but do not establish high conditional shift.
What does the EDA show?
The selected state-year datasets have usable sample counts, different observed income-positive rates, and available harmonized features.
What remains unknown?
Whether Florida is a mild-shift validation state and whether the hidden state pairs have high conditional shift. Those questions require model-based evidence beyond differences in overall label rates.