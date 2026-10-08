# VBench prompts

`VBench_full_info.json` and `prompts/prompts_per_dimension/` are from
[VBench](https://github.com/Vchitect/VBench) (Apache-2.0) and are redistributed unchanged.

`prompts/prompts_per_dimension_FINAL736/` is the prompt set the paper reports. Every line in it is
one of VBench's own GPT-enhanced prompts; what is ours is the choice of which variant to use for
each caption. The videos are still scored against the original VBench captions, so the benchmark is
unchanged — only the text handed to the generator differs, which VBench explicitly allows.
