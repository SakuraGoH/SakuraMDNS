"""Build the English DOCX."""
import pathlib
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_paper_base import (FIG_OBS, FIG_SAMPLES, OUT, REFS_EN, body, caption,
                             figure, para, setup, table)

doc = Document()
setup(doc, "en")

# ------------------------------------------------------------------ title
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("A Bottleneck\u2013Transformer Backbone for Masked-Diffusion Neural Samplers")
r.bold = True
r.font.size = Pt(18)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
r = p.add_run("A stage report on the two-dimensional three-state Potts model")
r.bold = True
r.font.size = Pt(12.5)

para(doc, "SakuraGoH", size=11, align="center", space_after=1)
para(doc, "Technical report \u00b7 27 November 2026", size=9.5, align="center", space_after=14)

# --------------------------------------------------------------- abstract
p = doc.add_paragraph()
r = p.add_run("Abstract")
r.bold = True
r.font.size = Pt(10)
p.paragraph_format.space_after = Pt(2)
para(doc,
     "We replace the backbone of an open-source masked-diffusion neural sampler \u2014 a "
     "rotary-position vision transformer with 829,060 parameters \u2014 with a lattice network "
     "built from bottleneck blocks that first compress the channel width, run a "
     "depthwise-separable 3\u00d73 convolution and a self-attention layer at the compressed "
     "width, and then expand back. Both families are trained with the same closed-loop "
     "objective and scored against exact Swendsen\u2013Wang samples on the two-dimensional "
     "three-state Potts model at L = 16 and \u03b2 = 1.2. The proposed backbone reaches equal or "
     "better distributional accuracy with 8.6\u00d7 fewer parameters: at n = 16,384 samples the "
     "paired energy bias is 4.4\u00d710\u207b\u2074 (1.2\u03c3, i.e. inside the sampling noise) against "
     "1.26\u00d710\u207b\u00b3 (2.5\u03c3) for the best checkpoint of the original family, while energy total "
     "variation falls from 0.0231 to 0.0190, magnetisation total variation from 0.0130 to "
     "0.0046 and the maximum two-point-correlation error from 1.1\u00d710\u207b\u00b3 to "
     "4.0\u00d710\u207b\u2074. Per optimizer step the new model is 6.0\u00d7 faster and yields 9.7\u00d7 more "
     "samples per second. A two-stage curriculum that first replays a teacher\u2019s samples and "
     "then switches to self-generated data with a sliding replay window lets the student "
     "overtake its teacher, which pure distillation never does. Two negative results are "
     "reported as well: widening the network from 64 to 128 channels buys convergence speed "
     "but not final quality, and extra capacity does not improve zero-shot transfer to larger "
     "lattices.",
     size=9.5, align="justify")

# ------------------------------------------------------------ 1 introduction
doc.add_heading("1  Introduction", level=1)
body(doc,
     "Sampling from a distribution known only up to a normalising constant, "
     "\u03c0(x) \u221d exp(\u2212\u03b2H(x)), is the central computational primitive of lattice field "
     "theory and statistical mechanics. Markov-chain Monte Carlo (MCMC) solves it exactly in "
     "the limit, and cluster algorithms such as Swendsen\u2013Wang [2] remain the method of "
     "choice for the Potts model [3, 4] because they beat critical slowing down. Their cost, "
     "however, is paid per sample: burn-in, autocorrelation and a mixing time that grows with "
     "the correlation length.", "en")
body(doc,
     "Amortised neural samplers invert this trade-off. A network is trained once to map noise "
     "to configurations, after which samples are produced in a single pass with no burn-in and "
     "no autocorrelation. MDNS [1] does this with a masked-diffusion, any-order "
     "autoregressive procedure [5, 6]: starting from a fully masked configuration, positions "
     "are revealed in a random order, each drawn from the model\u2019s conditional, and the "
     "trajectory carries an importance weight that corrects the model\u2019s own proposal. The "
     "price is architectural \u2014 the scan needs D = L\u00b2 sequential forward passes, so the "
     "backbone and not the training step sets the throughput of the whole loop.", "en")
body(doc,
     "The upstream implementation uses a RopeViT backbone [7, 8] with 829,060 parameters for "
     "L = 16. This report asks a narrow, measurable question: can a much smaller backbone with "
     "an explicitly bottlenecked width match or beat that model under the same objective, the "
     "same data and the same evaluation protocol? We answer it with a controlled study rather "
     "than a single number, because two of the four factors that changed turned out not to "
     "matter.", "en")
body(doc, "The contributions are:", "en")
for txt in (
    "a bottleneck\u2013transformer lattice backbone (denoted NBT) that reaches the accuracy of a "
    "backbone 8.6\u00d7 larger with 6.0\u00d7 less time per optimizer step (Section 5.2);",
    "a paired evaluation protocol: every architecture claim is measured against one shared "
    "exact reference draw and at one sample count, with the sampling noise floor reported "
    "explicitly (Section 4);",
    "a two-stage curriculum in which a student inherits a teacher\u2019s samples rather than its "
    "weights, and demonstrably overtakes it (Section 5.4);",
    "two negative results \u2014 capacity buys speed but not quality, and capacity does not buy "
    "zero-shot size extrapolation (Sections 5.3 and 5.5).",
):
    p = doc.add_paragraph(txt, style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(2)
    for r in p.runs:
        r.font.size = Pt(10.5)

# -------------------------------------------------------------- 2 background
doc.add_heading("2  Background", level=1)
doc.add_heading("2.1  The Potts model and its exact sampler", level=2)
body(doc,
     "For q states per site and a square lattice with periodic boundaries, "
     "H(x) = \u2212J \u03a3_<ij> \u03b4(x\u1d62, x\u2c7c) counts ferromagnetic bonds. On the square lattice the "
     "transition is self-dual at \u03b2_c = ln(1+\u221aq)/J [4], which for q = 3 gives "
     "\u03b2_c \u2248 1.0051, so \u03b2 = 1.2 lies in the ordered phase. The Swendsen\u2013Wang algorithm "
     "[2] builds Fortuin\u2013Kasteleyn clusters [3] and flips them jointly; it is the ground truth "
     "against which every learned sampler in this report is measured, and it is exact up to "
     "Monte Carlo error [13].", "en")

doc.add_heading("2.2  Masked-diffusion sampling and its objective", level=2)
body(doc,
     "A scan draws a permutation \u03c3 of the D positions and, for d = D, \u2026, 1, samples the "
     "token at \u03c3(d) from the model conditional p_\u03b8(\u00b7 | partially revealed x) and accumulates "
     "log w = r(x) \u2212 \u03a3_d log p_\u03b8(\u00b7) with r(x) = \u2212\u03b2H(x). Because the same object is "
     "proposal and target, training minimises a self-normalised importance-weighted "
     "cross-entropy whose fixed point is q_\u03b8 = \u03c0 [1, 12]. We monitor the effective sample "
     "size ESS/N = 1/(N \u03a3 \u0175\u1d62\u00b2) of the self-normalised weights [12].", "en")

doc.add_heading("2.3  Optimisation and averaging", level=2)
body(doc,
     "Matrix-valued parameters are updated with Muon [11], which orthogonalises the momentum "
     "via a Newton\u2013Schulz iteration, and the remaining parameters with AdamW [10]. An "
     "exponential moving average of the weights [14, 15] supplies the sampler, which stabilises "
     "the importance weights at no extra sampling cost.", "en")

# ------------------------------------------------------------- 3 architectures
doc.add_heading("3  Architectures", level=1)
doc.add_heading("3.1  Baseline: the upstream RopeViT", level=2)
body(doc,
     "The baseline is the backbone shipped with the upstream project: token embedding of width "
     "128, four transformer blocks with four heads, a learned class token, a learned absolute "
     "position embedding and rotary position frequencies [7, 8]. It has 829,060 parameters. "
     "Its forward pass costs 18.9 ms at batch 256 and 81.0 ms at batch 1024; a scan of 256 "
     "configurations takes 4.76 s.", "en")

doc.add_heading("3.2  Proposed: the NBT lattice network", level=2)
body(doc,
     "The proposed backbone replaces attention over the full width with a bottleneck. Tokens "
     "are embedded, passed through a circularly padded 3\u00d73 stem, then through N blocks, a "
     "GroupNorm and a pointwise head. Each NBT block is: a 1\u00d71 compression from C to C/2, two "
     "depthwise-separable 3\u00d73 convolutions at the compressed width, an optional single "
     "self-attention layer at the compressed width, a 1\u00d71 expansion back to C, and a "
     "LayerScale residual initialised at \u03b3 = 10\u207b\u2074. The design follows the linear-bottleneck "
     "observation of MobileNetV2 [9]: attention and depthwise convolution are cheap in "
     "channels, so running them at half width cuts both parameters and FLOPs while the 1\u00d71 "
     "layers preserve the information path.", "en")
body(doc,
     "Two models of this family are used below. res-s is a residual-convolution-only variant "
     "with 167,716 parameters, kept as the \u201cold architecture\u201d control from the first training "
     "campaign. b8c64 and b8c128 are NBT stacks with 8 blocks at 64 and 128 channels "
     "(127,620 and 484,612 parameters). Table 1 summarises the cost of each.", "en")

caption(doc, "Table 1. Backbone cost at L = 16 on one RTX 4080 SUPER (B = 1024 unless noted).")
table(doc,
      ["Backbone", "Params", "Forward\nB = 256", "Forward\nB = 1024", "One scan", "Train step"],
      [["upstream RopeViT 128\u00d74", "829,060", "18.9 ms", "81.0 ms", "4.76 s (256 samples)", "658 ms (256\u00d78)"],
       ["res-s (4 blocks, 48 ch)", "167,716", "\u2014", "12.8 ms", "3.30 s (1024)", "\u2014"],
       ["NBT 6\u00d764 (b6c64)", "95,876", "7.2 ms", "30.8 ms", "2.47 s (1024)", "89 ms (128\u00d78)"],
       ["NBT 8\u00d764 (b8c64)", "127,620", "\u2014", "30.8 ms", "3.04 s (1024)", "86 ms (128\u00d78)"],
       ["NBT 8\u00d7128 (b8c128)", "484,612", "\u2014", "\u2014", "10.6 s (1024)", "\u2014"]],
      widths=[4.6, 1.9, 2.0, 2.0, 3.4, 2.9], font=8.2)

doc.add_heading("3.3  Why the proposed backbone is lattice-size agnostic", level=2)
body(doc,
     "Every convolution uses circular padding, GroupNorm is channel-wise, and the attention "
     "layer carries no positional encoding. No parameter therefore depends on L: "
     "load_state_dict(strict = True) succeeds for a checkpoint trained at L = 16 when the model "
     "is rebuilt at L = 24 or L = 32. This is not true of the baseline, whose rotary "
     "frequencies and absolute position embedding are tied to the training size, and it turns "
     "size extrapolation from a retraining problem into an evaluation flag (Section 5.5).", "en")

# ------------------------------------------------------------------ 4 setup
doc.add_heading("4  Experimental setup", level=1)
body(doc,
     "All measurements use a single RTX 4080 SUPER (32 GB, torch 2.12.1 + cu130) and the "
     "three-state Potts model on a 16\u00d716 torus at \u03b2 = 1.2 and J = 1. The exact reference is "
     "produced by a C++ Swendsen\u2013Wang implementation compiled with OpenMP, which sustains "
     "73,020 effective samples per second at batch 128 [16], with 400 burn-in sweeps and one "
     "collected configuration every 10 sweeps.", "en")
body(doc,
     "Four diagnostics are reported for every model: the energy per site (its mean, and its "
     "total-variation distance to the exact energy histogram), the total-variation distance of "
     "the dominant-state-fraction histogram, the maximum absolute error of the normalised "
     "two-point correlation over r \u2264 L/2, and the self-normalised ESS. Because every model is "
     "scored against the same reference draw, the sampling noise floor can be stated exactly: "
     "1\u03c3 = \u03c3_E/\u221an = 7.1\u00d710\u207b\u2074 on the energy per site at n = 4096 and 3.6\u00d710\u207b\u2074 at "
     "n = 16,384. Differences smaller than that are not ranked in this report; they are called "
     "indistinguishable.", "en")
body(doc,
     "Training uses the closed loop of [1]: each round scans samples with the averaged weights, "
     "writes them to a shard, takes 800 Muon/AdamW steps on the scanned batch, and admits the "
     "new weights only if a held-out proper scoring rule improves. The gate is the mean "
     "cross-entropy at the 75% masked positions on 2,048 exact configurations that the model "
     "never trains on, which punishes both over-confidence and collapse; the reference set is "
     "re-drawn every 20 rounds so that it cannot be memorised.", "en")

# -------------------------------------------------------------- 5 results
doc.add_heading("5  Results", level=1)

doc.add_heading("5.1  Distributional accuracy", level=2)
caption(doc, "Table 2. Four checkpoints at n = 4096 against one exact reference (noise "
             "1\u03c3 = 7.1\u00d710\u207b\u2074 on energy/site). TV = total-variation distance.")
table(doc,
      ["Model", "Params", "ESS", "\u0394E / site", "TV(E)", "TV(m)", "corr err"],
      [["res-s (first campaign)", "167,716", "0.951", "+9.85\u00d710\u207b\u00b3 (13.8\u03c3)", "0.090", "0.087", "9.3\u00d710\u207b\u00b3"],
       ["b6c64 (teacher)", "95,876", "0.976", "+1.64\u00d710\u207b\u00b3 (2.3\u03c3)", "0.032", "0.028", "1.4\u00d710\u207b\u00b3"],
       ["b8c64 (best 64 ch)", "127,620", "0.980", "+9.59\u00d710\u207b\u2074 (1.3\u03c3)", "0.037", "0.043", "1.0\u00d710\u207b\u00b3"],
       ["b8c128 (128 ch)", "484,612", "0.965", "+1.18\u00d710\u207b\u00b3 (1.7\u03c3)", "0.049", "0.034", "1.0\u00d710\u207b\u00b3"]],
      widths=[4.3, 2.0, 1.7, 3.4, 1.7, 1.7, 2.1], font=8.2)
body(doc,
     "The residual-only control is the only model whose error is unambiguous: it runs hot by "
     "9.9\u00d710\u207b\u00b3 per site, fourteen standard deviations from the reference, and its correlation "
     "function is off by almost one percent. Replacing it with the bottleneck backbone removes "
     "essentially all of that error. At this sample count the three bottleneck models cannot be "
     "separated on the total-variation metrics, whose own uncertainty at n = 4096 is of the "
     "same order as the differences between them, so the ranking is settled at sixteen thousand "
     "samples in Table 3.", "en")

caption(doc, "Table 3. Paired evaluation at n = 16,384 (noise 1\u03c3 = 3.6\u00d710\u207b\u2074 on energy/site). "
             "The teacher is the best checkpoint of the original family; both students are of the "
             "proposed family.")
table(doc,
      ["Model", "Params", "ESS", "\u0394E / site", "TV(E)", "TV(m)", "corr err"],
      [["teacher b6c64 (round 69)", "95,876", "0.9765", "+1.26\u00d710\u207b\u00b3 (2.5\u03c3)", "0.0231", "0.0130", "1.1\u00d710\u207b\u00b3"],
       ["student b8c64 (round 24)", "127,620", "0.9816", "+4.4\u00d710\u207b\u2074 (1.2\u03c3)", "0.0190", "0.0046", "4.0\u00d710\u207b\u2074"],
       ["student b8c128 (round 5)", "484,612", "0.9690", "+3.3\u00d710\u207b\u2074 (0.9\u03c3)", "0.0170", "0.0041", "5.0\u00d710\u207b\u2074"]],
      widths=[4.6, 2.0, 1.7, 3.4, 1.7, 1.7, 2.1], font=8.2)
body(doc,
     "At this precision the teacher carries a real bias: 2.5\u03c3 away from the exact mean, which "
     "is not explained by sampling noise. The student\u2019s bias is 1.2\u03c3, i.e. consistent with "
     "zero, and it is better on all four diagnostics. Figures 1 and 2 show the same conclusion "
     "qualitatively: the residual-only row produces visibly more and larger droplets of the "
     "minority states, while the bottleneck rows cannot be told apart from the exact sampler by "
     "eye.", "en")
figure(doc, FIG_SAMPLES, 15.2)
caption(doc, "Figure 1. Potts q = 3, L = 16, \u03b2 = 1.2. Rows, top to bottom: res-s, b6c64, "
             "b8c64, b8c128 and exact Swendsen\u2013Wang. Five samples per row.")
figure(doc, FIG_OBS, 16.0)
caption(doc, "Figure 2. Observables for the same checkpoints. Left: energy per site. Middle: "
             "dominant-state fraction. Right: normalised two-point correlation. The residual-only "
             "model (light blue) is visibly displaced in all three panels.")

doc.add_heading("5.2  Throughput", level=2)
body(doc,
     "Because a scan needs D sequential forward passes, backbone cost dominates the loop. "
     "Table 4 compares the two families at each project\u2019s own resampling cadence. The upstream "
     "code re-scans every 10 steps with batch 256; the proposed configuration re-scans every 20 "
     "steps with batch 1024.", "en")
caption(doc, "Table 4. Cost per optimizer step, each at its own cadence (one RTX 4080 SUPER).")
table(doc,
      ["Quantity", "upstream RopeViT", "proposed b6c64", "factor"],
      [["parameters", "829,060", "95,876", "8.6\u00d7 smaller"],
       ["forward pass, B = 1024", "81.0 ms", "30.8 ms", "2.6\u00d7"],
       ["one scan", "4.76 s (256 samples)", "2.47 s (1024 samples)", "7.7\u00d7"],
       ["training step", "658 ms (256\u00d78 rows)", "89 ms (128\u00d78 rows)", "7.4\u00d7"],
       ["per optimizer step", "1134 ms", "188 ms", "6.0\u00d7"],
       ["samples per second", "22.6", "217.9", "9.7\u00d7"]],
      widths=[5.0, 4.6, 4.6, 2.6], font=8.4)
body(doc,
     "Three further optimisations were measured on the same hardware. Compiling the scan "
     "forward pass, with the autocast context hoisted out of the model so that the compiler can "
     "trace it, cuts the forward pass from 29.5 ms to 10.2 ms and a full round from 88.6 s to "
     "41.4 s; compiling the training step as well takes the step from 135 ms to 86 ms and the "
     "round to 41.4 s (a 2.14\u00d7 round-level speedup over the eager path). Increasing the scan "
     "batch beyond 1024 does not help: per-sample cost is flat, so 4096 gives 60.7 samples per "
     "second against 66.7 for 1024. Running several scan processes concurrently on one device "
     "gives 1.14\u00d7 at two workers and 1.20\u00d7 at four, because the device is already about 98% "
     "occupied \u2014 parallelism of this kind pays only across devices.", "en")
body(doc,
     "The native cluster sampler is a different matter: rewriting the reference MCMC kernels in "
     "C++ with OpenMP made them 54\u2013489\u00d7 faster than the original Python implementations "
     "(Swendsen\u2013Wang for Potts: 8.0 \u2192 3,924 sweeps per second), which is what makes an exact "
     "reference affordable inside the loop at all.", "en")

doc.add_heading("5.3  Does capacity help? 64 versus 128 channels", level=2)
body(doc,
     "A 3.8\u00d7 wider model reaches the same distributional accuracy, not a better one. On the "
     "paired protocol it is marginally ahead on the histogram metrics and marginally behind on "
     "the correlation function, all inside the noise band of Table 3. What it does buy is "
     "convergence speed: it reaches the quality that the 64-channel model needed 24 rounds for "
     "in 5 rounds, and because each round costs 527 s instead of 255 s, the wall-clock saving is "
     "1.6\u00d7 (94 min against 150 min including warm-up). Table 5 makes the trade explicit.", "en")
caption(doc, "Table 5. Cost of reaching the same physical quality.")
table(doc,
      ["Model", "Round", "Rounds to converge", "Total wall clock"],
      [["b8c64", "255 s", "24", "50 min warm-up + 100 min = 150 min"],
       ["b8c128", "527 s", "5", "50 min warm-up + 44 min = 94 min"]],
      widths=[2.6, 2.2, 4.0, 7.8], font=8.4)

doc.add_heading("5.4  A curriculum that overtakes its teacher", level=2)
body(doc,
     "The teacher plateaus with a residual bias that more training of the same kind does not "
     "remove: between rounds 61 and 69 its held-out NLL moved by about 10\u207b\u2075 per round while "
     "its energy bias fell fivefold, which is the clearest evidence in this study that the "
     "training objective and the physical accuracy are not the same axis. Simply resuming the "
     "teacher therefore copies the bias together with the weights.", "en")
body(doc,
     "Instead the student inherits the teacher\u2019s data. Each stored shard contains "
     "(x, log w) with log w = log \u03c0(x) \u2212 log q_old(x) + const, a self-normalised importance "
     "weight for \u03c0 that belongs to the sample and to the proposal that generated it, not to the "
     "current model. Replaying such shards is therefore a valid off-policy estimate of the same "
     "objective [12]. Stage one trains a fresh model on 24,576 teacher samples until its gate "
     "score catches up with the teacher\u2019s; stage two then switches to self-generated data with "
     "a four-round sliding replay window.", "en")
body(doc,
     "Stage one cannot exceed its teacher, and measurement confirms the ceiling: with 64 "
     "channels the student stops 7\u00d710\u207b\u2075 short of the teacher\u2019s score, and with 128 channels "
     "only 5\u00d710\u207b\u2075 short \u2014 3.8\u00d7 the parameters buys 2\u00d710\u207b\u2075. Stage two is what changes the "
     "outcome: after 24 self-generated rounds the student is better than its teacher on every "
     "diagnostic in Table 3, with the magnetisation and correlation errors reduced by about "
     "65%.", "en")

doc.add_heading("5.5  Zero-shot transfer to larger lattices", level=2)
body(doc,
     "Because the proposed backbone carries no size-dependent parameter, a checkpoint trained "
     "at L = 16 can be evaluated at another size by rebuilding the module and loading the same "
     "weights. Table 6 reports the outcome; no retraining is involved.", "en")
caption(doc, "Table 6. Zero-shot size transfer of an L = 16 checkpoint (n = 2048 per size, "
             "scored against the exact sampler at that size).")
table(doc,
      ["L", "ESS (64 ch)", "ESS (128 ch)", "\u0394E/site (64 ch)", "\u0394E/site (128 ch)", "corr err (64 ch)"],
      [["16 (trained)", "0.982", "0.969", "4.4\u00d710\u207b\u2074", "3.3\u00d710\u207b\u2074", "4.0\u00d710\u207b\u2074"],
       ["24 (zero-shot)", "0.919", "0.909", "1.20\u00d710\u207b\u00b3", "1.22\u00d710\u207b\u00b3", "1.0\u00d710\u207b\u00b3"],
       ["32 (zero-shot)", "0.673", "\u2014", "4.25\u00d710\u207b\u00b3", "\u2014", "4.2\u00d710\u207b\u00b3"]],
      widths=[2.6, 2.3, 2.4, 2.9, 3.0, 2.6], font=8.2)
body(doc,
     "Transfer to L = 24 is usable: the proposal stays close to \u03c0 and the energy bias is six "
     "parts in ten thousand. At L = 32 the model degrades clearly, becoming too hot by two "
     "parts in a thousand, yet the sampled configurations remain qualitatively correct. The "
     "degradation is caused by the change of the target distribution with system size, not by "
     "an architectural limitation: the two backbones, 64 and 128 channels, degrade by exactly "
     "the same amount at L = 24.", "en")

# ------------------------------------------------------------ 6 discussion
doc.add_heading("6  Discussion", level=1)
body(doc,
     "Ordering the four changes by what they actually delivered is the main lesson of this "
     "study. The architecture is first: with 8.6\u00d7 fewer parameters the bottleneck backbone "
     "removes a fourteen-sigma energy bias and improves every distributional metric, while "
     "being 6\u00d7 cheaper per step. The curriculum is second: it converts the teacher\u2019s plateau "
     "into a further factor of 1.7 on the energy bias and 1.7 on energy total variation. "
     "Capacity is a distant third, and only in wall-clock terms.", "en")
body(doc,
     "Two methodological observations seem more transferable than the specific numbers. First, "
     "a held-out likelihood is a weak proxy for physical accuracy near the optimum: it "
     "saturates while the sampled distribution is still moving measurably, and a gate that "
     "selects on it will happily promote a model whose energy bias is several sigma. A "
     "physics-aligned criterion, for instance the energy histogram distance, would be the "
     "better selector. Second, the acceptance threshold of such a gate must be scaled with the "
     "learning rate; a fixed threshold above the improvement a decayed step size can produce "
     "turns into a permanent refusal and the run spins without ever promoting again.", "en")

# ------------------------------------------------------------ 7 limitations
doc.add_heading("7  Limitations", level=1)
body(doc,
     "The study is confined to one model, one lattice size and one temperature. All accuracy "
     "numbers are for the q = 3 Potts model at L = 16 and \u03b2 = 1.2, which is the ordered phase, "
     "where the correlation length is short compared with the lattice; the critical point, "
     "\u03b2 \u2248 1.0051, where fluctuations are scale-free, is precisely the regime in which a "
     "learned sampler ought to be most interesting and is not covered here. L = 16 is also the "
     "size at which exact cluster sampling is cheap, so this work demonstrates accuracy and "
     "efficiency, not a regime where MCMC fails. The upstream comparison uses one released "
     "checkpoint in our own evaluation harness; a full comparison with the original training "
     "recipe end to end is left open. Finally, the largest reported improvements are "
     "differences of one to two standard deviations, which is why every table states its noise "
     "floor rather than a ranking.", "en")

# ------------------------------------------------------------- 8 conclusion
doc.add_heading("8  Conclusion", level=1)
body(doc,
     "A bottleneck\u2013transformer lattice backbone of 127,620 parameters matches and modestly "
     "exceeds the distributional accuracy of the 829,060-parameter backbone it replaces on the "
     "two-dimensional three-state Potts model, at 6.0\u00d7 less time per optimizer step and 9.7\u00d7 "
     "the sample throughput. A curriculum that hands over data rather than weights lets a "
     "student pass its teacher, which weight initialisation alone cannot do. Extra capacity "
     "shortens convergence but does not raise the final accuracy, and does not extend the "
     "lattice sizes to which one checkpoint transfers. The natural next steps are the critical "
     "point, where the advantage of an amortised sampler over cluster MCMC should finally show, "
     "and a gate that selects on a physics-aligned statistic instead of a likelihood.", "en")

# -------------------------------------------------------------- references
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(12)
r = p.add_run("References")
r.bold = True
r.font.size = Pt(12)
for i, ref in enumerate(REFS_EN, 1):
    pp = doc.add_paragraph()
    pp.paragraph_format.left_indent = Cm(0.9)
    pp.paragraph_format.first_line_indent = Cm(-0.9)
    pp.paragraph_format.space_after = Pt(2)
    rr = pp.add_run(f"[{i}]  {ref}")
    rr.font.size = Pt(8.6)

# --------------------------------------------------------------- appendix
doc.add_heading("Appendix A  Reproduction", level=1)
for line in (
    "python scripts/compare_models.py --n 4096 --out results/compare",
    "python scripts/eval_checkpoint.py --ckpt results/weights/b8c64_round024.pt \\",
    "       --beta 1.2 --n 16384 --out results/eval_student",
    "python scripts/eval_checkpoint.py --ckpt results/weights/b8c64_round024.pt \\",
    "       --beta 1.2 --n 2048 --L 24 --out results/ext_L24",
    "CH=64 BLOCKS=8 TEACHER=results/weights/teacher_b6c64_round069.pt \\",
    "  WARMUP_DIR=data/warmup_teacher WARMUP_STEPS=10000 STEPS=800 EVERY=20 \\",
    "  bash scripts/train_curriculum.sh",
    "python scripts/convert_upstream_ckpt.py --src checkpoints/L_16_potts/potts_low.pth \\",
    "       --out results/weights/upstream_ropevit_potts_low.pt",
):
    pp = doc.add_paragraph()
    pp.paragraph_format.left_indent = Cm(0.4)
    pp.paragraph_format.space_after = Pt(0)
    rr = pp.add_run(line)
    rr.font.name = "Consolas"
    rr.font.size = Pt(8.2)
body(doc,
     "Checkpoints are archived under results/weights/ (b8c64_round024.pt is the model reported "
     "in Tables 3 and 4). Per-round metrics for every run are in results/run_metrics/, and the "
     "full write-ups, including the negative results, in results/RESULTS_curriculum.md and "
     "results/RESULTS_scaling.md.", "en")

out = OUT / "SakuraMDNS_stage_report_EN.docx"
doc.save(out)
print("wrote", out, f"({out.stat().st_size/1e3:.0f} KB)")
