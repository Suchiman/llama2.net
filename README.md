# llama2.net

This is a pure C# port of Alfonso² Peterssen's [Java port](https://github.com/mukel/llama2.java) of Andrej Karpathy's awesome [llama2.c](https://github.com/karpathy/llama2.c), a very simple implementation
to run inference of models with a [Llama2](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.

## Build
Only needs the .NET 7 SDK.  
The code expects [`tokenizer.bin`](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) in the current directory.  
The sample `stories15M.bin` model can be found [here](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin)

To build and run:
```bash
dotnet run -c Release stories15M.bin
```
