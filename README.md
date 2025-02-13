<h1>Neural Image Compressor</h1>

<p>This repository contains the implementation of a deep learning-based image compression model that achieves state-of-the-art rate-distortion performance. The project leverages a variational autoencoder architecture combined with non-local attention mechanisms to provide efficient, high-quality image compression.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#training">Training</a></li>
  <li><a href="#citation">Citation</a></li>
  <li><a href="#license">License</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>This project implements a neural image compressor using modern deep learning techniques for lossy image compression. The model is built on a variational autoencoder framework and employs non-local attention to capture global context, resulting in high compression ratios without sacrificing image quality.</p>

<h2 id="features">Features</h2>
<ul>
  <li><strong>End-to-End Learned Compression:</strong> The model learns the entire compression process directly from data.</li>
  <li><strong>Non-Local Attention Mechanism:</strong> Enhances the model's ability to capture global image context for improved compression.</li>
  <li><strong>Improved Entropy Model:</strong> Utilizes a 3D CNN-based autoregressive prior to achieve accurate bitrate estimation.</li>
  <li><strong>Variable Rate Compression:</strong> Supports multiple compression rates with a single model instance.</li>
  <li><strong>State-of-the-Art Performance:</strong> Outperforms traditional codecs (JPEG, JPEG2000, BPG) on benchmark datasets.</li>
  <li><strong>Efficient Inference:</strong> Designed for fast and practical deployment in real-world applications.</li>
</ul>

<h2 id="installation">Installation</h2>
<ol>
  <li>Clone the repository:
    <pre><code>git clone https://github.com/username/neural-image-compressor.git
cd neural-image-compressor</code></pre>
  </li>
  <li>Install the required dependencies:
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
</ol>

<h2 id="usage">Usage</h2>
<p>To compress an image:</p>
<pre><code>from compressor import NeuralCompressor

model = NeuralCompressor()
compressed = model.compress("input.png")
compressed.save("compressed.bin")</code></pre>
<p>To decompress an image:</p>
<pre><code>decompressed = model.decompress("compressed.bin")
decompressed.save("output.png")</code></pre>

<h2 id="results">Results</h2>
<p>Our model achieves the following performance on the Kodak dataset:</p>
<table>
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>PSNR</td>
    <td>32.4 dB</td>
  </tr>
  <tr>
    <td>MS-SSIM</td>
    <td>0.978</td>
  </tr>
  <tr>
    <td>BPP</td>
    <td>0.15</td>
  </tr>
</table>

<h2 id="training">Training</h2>
<p>To train the model on your own dataset, execute the following command:</p>
<pre><code>python train.py --data_dir /path/to/images --epochs 100</code></pre>

<h2 id="citation">Citation</h2>
<p>If you use this code for your research, please cite our paper:</p>
<pre><code>@article{author2023neural,
  title={Neural Image Compression via Non-Local Attention},
  author={Author, A. and Author, B.},
  journal={arXiv preprint arXiv:2301.00000},
  year={2023}
}</code></pre>

<h2 id="license">License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for further details.</p>
