# Singing-Voice-Separation
This is an implementation of U-Net for vocal separation with tensorflow

## Requirement
- librosa==0.6.2
- numpy==1.14.3
- tensorflow==1.9.0
- python==3.6.5

## Data
I prepare CCMixter datasets in "./data" and Each track consisted of Mixed, instrumental, Vocal version
<pre><code>$ python CCMixter_process.py</code></pre>

## Usage
- Train
<pre><code>$ python Training.py</code></pre>
- Test
<pre><code>$ python Test.py</code></pre>

## Paper
Andreas Jansson, et al. SINGING VOICE SEPARATION WITH DEEP U-NET CONVOLUTIONAL NETWORKS. 2017. <br> paper: https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf
