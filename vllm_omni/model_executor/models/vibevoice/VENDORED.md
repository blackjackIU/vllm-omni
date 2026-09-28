# Vendored VibeVoice inference components

`vendored_tokenizer.py` and `vendored_dpm_solver.py` are derived from
`vibevoice-community/VibeVoice` commit
`631804b9c1f042e381207fe87c54603fe6accbc1`, which preserves Microsoft's
VibeVoice 1.5B inference implementation.

Only inference-time codec and scheduler code is included. The tokenizer is
kept numerically aligned with upstream, except that its optional Apex branch
is replaced by the native RMSNorm path that upstream uses when
`OPTIMIZE_FOR_SPEED=0`. AutoModel registration is intentionally omitted so
importing vLLM-Omni does not mutate Transformers' global model mappings.

The tokenizer source is MIT licensed, Copyright (c) 2025 Microsoft.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The DPM solver retains its upstream Apache-2.0 header and copyright notice in
the file.
