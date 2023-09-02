using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text;

readonly struct BigMemory<T> where T : unmanaged
{
    private readonly unsafe byte* _ptr;
    private readonly long _start;
    private readonly long _length;

    public unsafe BigMemory(MemoryMappedViewAccessor view, long start, long length)
    {
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref _ptr);
        _start = start;
        _length = length;
    }

    private unsafe BigMemory(byte* ptr, long start, long length)
    {
        _ptr = ptr;
        _start = start;
        _length = length;
    }

    public Span<T> Span => GetSpan();

    public unsafe T* Pointer => (T*)(_ptr + _start);

    public unsafe Span<T> GetSpan()
    {
        return new Span<T>(_ptr + _start, (int)_length);
    }

    public unsafe BigMemory<T> Slice(long start)
    {
        if ((ulong)start > (ulong)_length)
            throw new ArgumentOutOfRangeException(nameof(start));

        return new BigMemory<T>(_ptr, _start + start, _length - start);
    }

    public unsafe BigMemory<T> Slice(long start, long length)
    {
        if ((ulong)start > (ulong)_length || (ulong)length > (ulong)(_length - start))
            throw new ArgumentOutOfRangeException();

        return new BigMemory<T>(_ptr, _start + start, length);
    }
}

sealed class Config
{
    public readonly int dim; // transformer dimension
    public readonly int hidden_dim; // for ffn layers
    public readonly int n_layers; // number of layers
    public readonly int n_heads; // number of query heads
    public readonly int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    public readonly int vocab_size; // vocabulary size, usually 256 (byte-level)
    public readonly int seq_len; // max sequence length
    public readonly bool shared_weights;
    public readonly int head_size;

    public Config(BinaryReader buffer)
    {
        this.dim = buffer.ReadInt32();
        this.hidden_dim = buffer.ReadInt32();
        this.n_layers = buffer.ReadInt32();
        this.n_heads = buffer.ReadInt32();
        this.n_kv_heads = buffer.ReadInt32();
        int vocab_size = buffer.ReadInt32();
        this.vocab_size = Math.Abs(vocab_size);
        this.seq_len = buffer.ReadInt32();
        this.shared_weights = vocab_size > 0;
        this.head_size = dim / n_heads;
    }

    public override string ToString()
    {
        return "Config{" +
                "dim=" + dim +
                ", hidden_dim=" + hidden_dim +
                ", n_layers=" + n_layers +
                ", n_heads=" + n_heads +
                ", n_kv_heads=" + n_kv_heads +
                ", vocab_size=" + vocab_size +
                ", seq_len=" + seq_len +
                ", shared_weights=" + shared_weights +
                ", head_size=" + head_size +
                '}';
    }
}

sealed class Weights
{
    // token embedding table
    public readonly BigMemory<float> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public readonly BigMemory<float>[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls. note dim == n_heads * head_size
    public readonly BigMemory<float>[] wq; // (layer, dim, n_heads * head_size)
    public readonly BigMemory<float>[] wk; // (layer, dim, n_kv_heads * head_size)
    public readonly BigMemory<float>[] wv; // (layer, dim, n_kv_heads * head_size)
    public readonly BigMemory<float>[] wo; // (layer, n_heads * head_size, dim)
    public readonly BigMemory<float>[] rms_ffn_weight; // (layer, dim)
    // weights for ffn
    public readonly BigMemory<float>[] w1; // (layer, hidden_dim, dim)
    public readonly BigMemory<float>[] w2; // (layer, dim, hidden_dim)
    public readonly BigMemory<float>[] w3; // (layer, hidden_dim, dim)
    // readonly rmsnorm
    public readonly BigMemory<float> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    public readonly BigMemory<float> wcls; // (vocab_size, dim)

    static BigMemory<float> takeFloats(BigMemory<float> memorySegment, long[] position, params int[] dims)
    {
        long totalBytes = 1;
        foreach (int d in dims)
        {
            totalBytes *= d;
        }
        totalBytes *= sizeof(float);
        var slice = memorySegment.Slice(position[0], totalBytes);
        position[0] += totalBytes;
        return slice;
    }

    static BigMemory<float>[] takeArray(BigMemory<float> memorySegment, long[] position, int dim0, params int[] dims)
    {
        BigMemory<float>[] segments = new BigMemory<float>[dim0];
        for (int i = 0; i < dim0; ++i)
        {
            segments[i] = takeFloats(memorySegment, position, dims);
        }
        return segments;
    }

    // ----------------------------------------------------------------------------
    // initialization: read from checkpoint

    public Weights(Config config, BigMemory<float> memorySegment)
    {
        long[] position = new long[] { 0 };
        this.token_embedding_table = takeFloats(memorySegment, position, config.vocab_size, config.dim);
        this.rms_att_weight = takeArray(memorySegment, position, config.n_layers, config.dim);
        this.wq = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_heads * config.head_size);
        this.wk = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wv = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wo = takeArray(memorySegment, position, config.n_layers, config.n_heads * config.head_size, config.dim);
        this.rms_ffn_weight = takeArray(memorySegment, position, config.n_layers, config.dim);
        this.w1 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.w2 = takeArray(memorySegment, position, config.n_layers, config.dim, config.hidden_dim);
        this.w3 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.rms_final_weight = takeFloats(memorySegment, position, config.dim);
        position[0] += (config.seq_len * config.head_size / 2) * sizeof(float); // skip what used to be freq_cis_real (for RoPE)
        position[0] += (config.seq_len * config.head_size / 2) * sizeof(float); // skip what used to be freq_cis_imag (for RoPE)
        this.wcls = config.shared_weights
                ? this.token_embedding_table
                : takeFloats(memorySegment, position, config.vocab_size, config.dim);
    }
}

sealed class RunState
{
    // current wave of activations
    public readonly float[] x; // activation at current time stamp (dim,)
    public readonly float[] xb; // same, but inside a residual branch (dim,)
    public readonly float[] xb2; // an additional buffer just for convenience (dim,)
    public readonly float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public readonly float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public readonly float[] q; // query (dim,)
    public readonly float[] k; // key (dim,)
    public readonly float[] v; // value (dim,)
    public readonly float[] att; // buffer for scores/attention values (n_heads, seq_len)
    public readonly float[] logits; // output logits
    // kv cache
    public readonly float[] key_cache;   // (layer, seq_len, dim)
    public readonly float[] value_cache; // (layer, seq_len, dim)

    public RunState(Config config)
    {
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        this.x = new float[config.dim];
        this.xb = new float[config.dim];
        this.xb2 = new float[config.dim];
        this.hb = new float[config.hidden_dim];
        this.hb2 = new float[config.hidden_dim];
        this.q = new float[config.dim];
        this.k = new float[kv_dim];
        this.v = new float[kv_dim];
        this.att = new float[config.n_heads * config.seq_len];
        this.logits = new float[config.vocab_size];
        this.key_cache = new float[config.n_layers * config.seq_len * kv_dim];
        this.value_cache = new float[config.n_layers * config.seq_len * kv_dim];
    }
}

sealed class Transformer
{
    public readonly Config config; // the hyperparameters of the architecture (the blueprint)
    public readonly Weights weights; // the weights of the model
    public readonly RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    private readonly MemoryMappedFile mappedFile;

    public Transformer(string checkpoint_path)
    {
        this.mappedFile = MemoryMappedFile.CreateFromFile(checkpoint_path);
        int configSize = 7 * sizeof(int);
        // read in the config header
        using var configBuffer = mappedFile.CreateViewStream(0, configSize, MemoryMappedFileAccess.Read);
        var configReader = new BinaryReader(configBuffer, Encoding.UTF8);
        this.config = new Config(configReader);
        Console.WriteLine(config);
        this.state = new RunState(config);
        var accessor = mappedFile.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        this.weights = new Weights(config, new BigMemory<float>(accessor, configSize, accessor.Capacity));
    }
}

sealed class Tokenizer
{
    public readonly string[] vocab;
    public readonly float[] vocab_scores;
    public readonly int vocab_size;
    public readonly int max_token_length;
    public Dictionary<string, int>? sorted_vocab;

    public Tokenizer(string tokenizer_path, int vocab_size)
    {
        // i should have written the vocab_size into the tokenizer file... sigh
        this.vocab_size = vocab_size;
        // malloc space to hold the scores and the strings
        this.vocab = new string[vocab_size];
        this.vocab_scores = new float[vocab_size];

        // read in the file
        using var map = MemoryMappedFile.CreateFromFile(tokenizer_path);
        using var accessor = map.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        var reader = new BinaryReader(accessor, Encoding.UTF8);
        this.max_token_length = reader.ReadInt32();
        for (int i = 0; i < vocab_size; i++)
        {
            this.vocab_scores[i] = reader.ReadSingle();
            int len = reader.ReadInt32();
            byte[] bytes = reader.ReadBytes(len);
            this.vocab[i] = Encoding.UTF8.GetString(bytes);
        }
    }
}

sealed class Sampler
{
    public readonly int vocab_size;
    public readonly int[] probindex; // buffer used in top-p sampling
    public readonly float temperature;
    public readonly float topp;
    public long rng_seed;

    public Sampler(int vocab_size, float temperature, float topp, long rng_seed)
    {
        this.vocab_size = vocab_size;
        this.temperature = temperature;
        this.topp = topp;
        this.rng_seed = rng_seed;
        // buffer only used with nucleus sampling; may not need but it's ~small
        this.probindex = new int[vocab_size];
    }

    int random_u32()
    {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        rng_seed ^= rng_seed >> 12;
        rng_seed ^= rng_seed << 25;
        rng_seed ^= rng_seed >> 27;
        return (int)((rng_seed * 0x2545F4914F6CDD1DL) >> 32);
    }

    public float random_f32()
    { // random float32 in [0,1)
        return (random_u32() >>> 8) / 16777216.0f;
    }
}

class Llama2
{
    // ----------------------------------------------------------------------------
    // neural net blocks; the dynamics of the Transformer

    static void rmsnorm(float[] o, float[] x, BigMemory<float> weight, int size)
    {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++)
        {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float)Math.Sqrt(ss);
        var w = weight.Span;
        // normalize and scale
        for (int j = 0; j < size; j++)
        {
            o[j] = w[j] * (ss * x[j]);
        }
    }

    static void softmax(float[] x, int xOffset, int size)
    {
        // find max value (for numerical stability)
        float max_val = x[0 + xOffset];
        for (int i = 1; i < size; i++)
        {
            if (x[i + xOffset] > max_val)
            {
                max_val = x[i + xOffset];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[i + xOffset] = (float)Math.Exp(x[i + xOffset] - max_val);
            sum += x[i + xOffset];
        }
        // normalize
        for (int i = 0; i < size; i++)
        {
            x[i + xOffset] /= sum;
        }
    }

    static unsafe void matmul(float[] xout, float[] x, BigMemory<float> wSegment, int n, int d)
    {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        Parallel.For(0, d, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, i =>
        {
            float val = 0f;
            var wp = wSegment.Pointer;
            fixed (float* xp = x)
            {
                val = matmul_simd((nuint)n, (nuint)i, wp, xp);
            }
            xout[i] = val;
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float matmul_simd(nuint n, nuint i, float* wp, float* xp)
    {
        Vector<float> sum0 = Vector<float>.Zero;
        Vector<float> sum1 = Vector<float>.Zero;
        Vector<float> sum2 = Vector<float>.Zero;
        Vector<float> sum3 = Vector<float>.Zero;
        nuint width = (nuint)Vector<float>.Count;
        nuint upperBound = n - n % (4 * width);
        nuint j = 0;

        for (; j < upperBound; j += 4 * width)
        {
            // In .NET 8, we actually want to just use V256, it perfectly unrolls to V128x2 for NEON or SSE2
            var wj0 = Unsafe.Read<Vector<float>>(&wp[i * n + j + 0 * width]);
            var wj1 = Unsafe.Read<Vector<float>>(&wp[i * n + j + 1 * width]);
            var wj2 = Unsafe.Read<Vector<float>>(&wp[i * n + j + 2 * width]);
            var wj3 = Unsafe.Read<Vector<float>>(&wp[i * n + j + 3 * width]);
            var xj1 = Unsafe.Read<Vector<float>>(&xp[j + 1 * width]);
            var xj0 = Unsafe.Read<Vector<float>>(&xp[j + 0 * width]);
            var xj2 = Unsafe.Read<Vector<float>>(&xp[j + 2 * width]);
            var xj3 = Unsafe.Read<Vector<float>>(&xp[j + 3 * width]);
            sum0 = fma(wj0, xj0, sum0);
            sum1 = fma(wj1, xj1, sum1);
            sum2 = fma(wj2, xj2, sum2);
            sum3 = fma(wj3, xj3, sum3);
        }
        float val = Vector.Sum(sum0 + sum1 + sum2 + sum3);
        for (; j < n; j++)
            val += wp[i * n + j] * xp[j];
        return val;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static Vector<float> fma(Vector<float> a, Vector<float> b, Vector<float> c)
        {
            if (Fma.IsSupported && Vector<float>.Count is 8)
            {
                // (a * b) + c
                return Fma.MultiplyAdd(
                    a.AsVector256(),
                    b.AsVector256(),
                    c.AsVector256()).AsVector();
            }
            else if (AdvSimd.IsSupported && Vector<float>.Count is 4)
            {
                // c + (a * b)
                return AdvSimd.FusedMultiplyAdd(
                    c.AsVector128(),
                    a.AsVector128(),
                    b.AsVector128()).AsVector();
            }
            else
            {
                return (a * b) + c;
            }
        }
    }

    static float[] forward(Transformer transformer, int token, int pos)
    {
        // a few convenience variables
        Config p = transformer.config;
        Weights w = transformer.weights;
        RunState s = transformer.state;
        int dim = p.dim;
        int hidden_dim = p.hidden_dim;
        int head_size = p.head_size;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        // copy the token embedding into x
        w.token_embedding_table.Span.Slice(token * dim, dim).CopyTo(s.x);

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++)
        {

            // attention rmsnorm
            rmsnorm(s.xb, s.x, w.rms_att_weight[l], dim);

            // qkv matmuls for this position
            matmul(s.q, s.xb, w.wq[l], dim, dim);
            matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
            matmul(s.v, s.xb, w.wv[l], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2)
            {
                int head_dim = i % head_size;
                float freq = (float)(1.0 / Math.Pow(10000.0f, head_dim / (float)head_size));
                float val = pos * freq;
                float fcr = (float)Math.Cos(val);
                float fci = (float)Math.Sin(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++)
                {
                    float[] vec = v == 0 ? s.q : s.k; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            Array.Copy(s.k, 0, s.key_cache, loff + pos * kv_dim, kv_dim);
            Array.Copy(s.v, 0, s.value_cache, loff + pos * kv_dim, kv_dim);

            // multihead attention. iterate over all heads
            Parallel.For(0, p.n_heads, h =>
            {
                // get the query vector for this head
                // float* q = s.q + h * head_size;
                int qOffset = h * head_size;

                // attention scores for this head
                // float* att = s.att + h * p.seq_len;
                int attOffset = h * p.seq_len;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++)
                {
                    // get the key vector for this head and at this timestep
                    // float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyCacheOffset = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++)
                    {
                        score += s.q[qOffset + i] * s.key_cache[keyCacheOffset + i];
                    }
                    score /= (float)Math.Sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attOffset + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * head_size;
                int xbOffset = h * head_size;
                // memset(xb, 0, head_size * sizeof(float));
                Array.Fill(s.xb, 0f, xbOffset, head_size);

                for (int t = 0; t <= pos; t++)
                {
                    // get the value vector for this head and at this timestep
                    // float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int vOffset = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attOffset + t];
                    // accumulate the weighted value inconfigto xb
                    for (int i = 0; i < head_size; i++)
                    {
                        s.xb[xbOffset + i] += a * s.value_cache[vOffset + i];
                    }
                }
            });

            // final matmul to get the output of the attention
            matmul(s.xb2, s.xb, w.wo[l], dim, dim);

            // residual connection back into x
            for (int i = 0; i < dim; i++)
            {
                s.x[i] += s.xb2[i];
            }

            // ffn rmsnorm
            rmsnorm(s.xb, s.x, w.rms_ffn_weight[l], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, w.w1[l], dim, p.hidden_dim);
            matmul(s.hb2, s.xb, w.w3[l], dim, p.hidden_dim);

            // SwiGLU non-linearity
            for (int i = 0; i < hidden_dim; i++)
            {
                float val = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (float)(1.0f / (1.0f + Math.Exp(-val)));
                // elementwise multiply with w3(x)
                s.hb[i] = val;
            }

            // elementwise multiply with w3(x)
            for (int i = 0; i < hidden_dim; i++)
            {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            matmul(s.xb, s.hb, w.w2[l], p.hidden_dim, dim);

            // residual connection
            for (int i = 0; i < dim; i++)
            {
                s.x[i] += s.xb[i];
            }
        }

        // final rmsnorm
        rmsnorm(s.x, s.x, w.rms_final_weight, dim);

        // classifier into logits
        matmul(s.logits, s.x, w.wcls, dim, p.vocab_size);
        return s.logits;
    }

    // ----------------------------------------------------------------------------
    // The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
    static ReadOnlySpan<char> decode(Tokenizer t, int prev_token, int token)
    {
        ReadOnlySpan<char> piece = t.vocab[token];
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if (prev_token == 1 && piece[0] == ' ') { piece = piece[1..]; }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'

        if (piece is ['<', '0', 'x', .. { } hex2, '>'])
        {
            char ch = (char)Int32.Parse(hex2, NumberStyles.HexNumber);
            // ok this token is a raw byte token, carefuly to only print printable chars or whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            bool isPrintable = (32 <= ch && ch < 127);
            if (isPrintable || char.IsWhiteSpace(ch))
            {
                piece = ch.ToString();
            }
        }
        return piece;
    }

    static int str_lookup(string str, Dictionary<string, int> sorted_vocab)
    {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        return sorted_vocab.GetValueOrDefault(str, -1);
    }

    static int encode(Tokenizer t, string text, int[] tokens)
    {
        // encode the string text (input) into an upper-bound preallocated tokens[] array

        if (t.sorted_vocab == null)
        {
            // sort vocabulary
            t.sorted_vocab = new Dictionary<string, int>();
            for (int i = 0; i < t.vocab_size; i++)
            {
                t.sorted_vocab.Add(t.vocab[i], i);
            }
        }

        // add_dummy_prefix is true by default
        tokens[0] = str_lookup(" ", t.sorted_vocab);
        int n_tokens = 1; // the number of tokens

        Span<byte> bytes = stackalloc byte[4];
        // first encode every individual codepoint in the input string
        foreach (var singleCodepoint in text.EnumerateRunes())
        {
            int id = str_lookup(singleCodepoint.ToString(), t.sorted_vocab);

            if (id != -1)
            {
                // we found this codepoint in vocab, add it as a token
                tokens[n_tokens++] = id;
            }
            else
            {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                int usedBytes = singleCodepoint.EncodeToUtf8(bytes);
                for (int i = 0; i < usedBytes; i++)
                {
                    tokens[n_tokens++] = bytes[i] + 3;
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true)
        {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; ++i)
            {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                string str_buffer = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]];
                int id = str_lookup(str_buffer, t.sorted_vocab);
                if (id != -1 && t.vocab_scores[id] > best_score)
                {
                    // this merge pair exists in vocab! record its score and position
                    best_score = t.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1)
            {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx + 1; i < n_tokens - 1; i++)
            {
                tokens[i] = tokens[i + 1];
            }
            n_tokens--; // token length decreased
        }

        return n_tokens;
    }

    // ----------------------------------------------------------------------------
    // utilities: time / rng
    static long time_in_ms()
    {
        // return time in milliseconds, for benchmarking the model speed
        return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
    }

    // ----------------------------------------------------------------------------
    // generation loop
    static void generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler, string? prompt, int steps)
    {
        // encode the (string) prompt into tokens sequence, if any is given
        int[]? prompt_tokens = null; // the sequence of prompt tokens
        int num_prompt_tokens = 0; // the total number of prompt tokens
        if (prompt != null)
        {
            prompt_tokens = new int[prompt.Length * 2];
            num_prompt_tokens = encode(tokenizer, prompt, prompt_tokens);
        }

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;     // position in the sequence
        while (pos < steps)
        {
            // forward the transformer to get logits for the next token
            float[] logits = forward(transformer, token, pos);

            // advance the state machine
            if (pos < num_prompt_tokens)
            {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens![pos];
            }
            else
            {
                // otherwise sample the next token from the logits
                next = sample(sampler, logits);
            }
            pos++;

            // data-dependent terminating condition: the BOS (1) token delimits sequences
            if (next == 1)
            {
                break;
            }

            // print the token as string, decode it with the Tokenizer object
            var piece = decode(tokenizer, token, next);
            Console.Out.Write(piece);

            token = next;

            // init the timer here because the first iteration can be slower
            if (start == 0)
            {
                start = time_in_ms();
            }
        }

        Console.WriteLine();

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1)
        {
            long end = time_in_ms();
            Console.Error.WriteLine("\nachieved tok/s: {0}", (pos - 1) / (double)(end - start) * 1000);
        }
    }

    // ----------------------------------------------------------------------------
    // sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
    static int sample_argmax(float[] probabilities, int n)
    {
        // return the index that has the highest probability
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < n; i++)
        {
            if (probabilities[i] > max_p)
            {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        return max_i;
    }

    static int sample_mult(float[] probabilities, int n, float coin)
    {
        // sample index from probabilities (they must sum to 1!)
        float cdf = 0.0f;
        for (int i = 0; i < n; i++)
        {
            cdf += probabilities[i];
            if (coin < cdf)
            {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    static void swap(int[] array, int from, int to)
    {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparison<int> comparator)
    {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n)
        {
            int r = 2 * prev + 2;
            if (r < n && comparator(array[r], array[next]) < 0)
            {
                next = r;
            }
            if (comparator(array[next], array[prev]) < 0)
            {
                swap(array, prev, next);
                prev = next;
            }
            else
            {
                break;
            }
        }
    }

    static int sample_topp(float[] probabilities, int n, float topp, int[] indices, float coin)
    {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()
        Comparison<int> comparator = (i, j) => (int)(probabilities[j] - probabilities[i]);

        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.Length; i++)
        {
            if (probabilities[i] >= cutoff)
            {
                indices[head++] = i;
            }
            else
            {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        // build heap O(n0)
        for (int i = n0 / 2 - 1; i >= 0; --i)
        {
            siftDown(indices, i, n0, comparator);
        }

        // truncate the list where cumulative probability of the largest k elements exceeds topp
        // O(k lg n0)
        float cumulative_prob = 0.0f;
        int last_idx = 0;
        for (int i = n0 - 1; i >= 0; i--)
        {
            swap(indices, 0, i);
            cumulative_prob += probabilities[indices[i]];
            if (cumulative_prob > topp)
            {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        // sample from the truncated list
        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= last_idx; i--)
        {
            cdf += probabilities[indices[i]];
            if (r < cdf)
            {
                return indices[i];
            }
        }

        return indices[last_idx]; // in case of rounding errors
    }

    static int sample(Sampler sampler, float[] logits)
    {
        // sample the token given the logits and some hyperparameters
        int next;
        if (sampler.temperature == 0.0f)
        {
            // greedy argmax sampling: take the token with the highest probability
            next = sample_argmax(logits, sampler.vocab_size);
        }
        else
        {
            // apply the temperature to the logits
            for (int q = 0; q < sampler.vocab_size; q++)
            {
                logits[q] /= sampler.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(logits, 0, sampler.vocab_size);
            // flip a (float) coin (this is our source of entropy for sampling)
            float coin = sampler.random_f32();
            // we sample from this distribution to get the next token
            if (sampler.topp <= 0 || sampler.topp >= 1)
            {
                // simply sample from the predicted probability distribution
                next = sample_mult(logits, sampler.vocab_size, coin);
            }
            else
            {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
            }
        }
        return next;
    }

    // ----------------------------------------------------------------------------
    // int main
    static void error_usage()
    {
        Console.WriteLine("Usage:   Llama2.net.exe <checkpoint> [options]");
        Console.WriteLine("Example: Lamma2.net.exe model.bin -n 256 -i \"Once upon a time\"");
        Console.WriteLine("Options:");
        Console.WriteLine("  -t <float>  temperature in [0,inf], default 1.0");
        Console.WriteLine("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9");
        Console.WriteLine("  -s <int>    random seed, default time(NULL)");
        Console.WriteLine("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
        Console.WriteLine("  -i <string> input prompt");
        Console.WriteLine("  -z <string> optional path to custom tokenizer");
    }

    public static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            error_usage();
            return;
        }

        // default parameters
        string checkpoint_path = args[0]; // e.g. out/model.bin
        string tokenizer_path = "tokenizer.bin";
        float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
        float topp = 0.9f;        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
        long rng_seed = 0;        // seed rng with time by default
        int steps = 256;          // max number of steps to run for, 0: use seq_len
        string? prompt = null;     // prompt string

        for (int i = 1; i < args.Length; i += 2)
        {
            // do some basic validation
            if (i + 1 >= args.Length) { error_usage(); } // must have arg after flag
            if (args[i][0] != '-') { error_usage(); } // must start with dash
            if (args[i].Length != 2) { error_usage(); } // must be -x (one dash, one letter)
            // read in the args
            switch (args[i][1])
            {
                case 't': temperature = float.Parse(args[i + 1]); break;
                case 'p': topp = float.Parse(args[i + 1]); break;
                case 's': rng_seed = int.Parse(args[i + 1]); break;
                case 'n': steps = int.Parse(args[i + 1]); break;
                case 'i': prompt = args[i + 1]; break;
                case 'z': tokenizer_path = args[i + 1]; break;
                default: error_usage(); break;
            }
        }

        // parameter validation/overrides
        if (rng_seed <= 0)
        {
            rng_seed = time_in_ms();
        }
        if (temperature < 0.0)
        {
            temperature = 0.0f;
        }
        if (topp < 0.0 || 1.0 < topp)
        {
            topp = 0.9f;
        }
        if (steps <= 0)
        {
            steps = 0;
        }

        // build the Transformer via the model .bin file
        Transformer transformer = new Transformer(checkpoint_path);
        if (steps == 0)
        {
            steps = transformer.config.seq_len; // ovrerride to ~max length
        }

        // build the Tokenizer via the tokenizer .bin file
        Tokenizer tokenizer = new Tokenizer(tokenizer_path, transformer.config.vocab_size);

        // build the Sampler
        Sampler sampler = new Sampler(transformer.config.vocab_size, temperature, topp, rng_seed);

        // run!
        generate(transformer, tokenizer, sampler, prompt, steps);
    }
}
