Creating a new, creative paraphraser focused on diverse generation. 

The paraphraser is specialized on text-to-text, meaningful paraphrase tasks, and could be utilized in generating diverse paraphrased sentences that convey the same meaning, with entirely rewording the original input. 

This differs from existing tools (e.g. Quillbot) in the sense that, it does not reword or change specific parts, but it instead rewrites the whole sentence in a creative way, like a human would do. 

The tool is at the moment based on T5-large, and will be trained on the PPDB paraphrasing dataset found here: http://paraphrase.org/#/download

The T5 model is being trained with respect to a specialized error function which, in addition to Cross Entropy Loss also takes into account the chrF metric which is used to measure how diverse the output is, and encourage the model to generate structurally different sentences.

Later, I plan on scaling the dataset more, including datasets like PAWS, Quora Question Answering dataset, and many more. In case I do, I plan on sharing that dataset publicly (with credit to originals) on GitHub and HuggingFace. 

I might also add better loss functions and additional metrics that promote creativity.