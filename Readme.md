<h1 class="code-line" data-line-start=0 data-line-end=1 ><a id="Siemens_Healthineers___ECG_Based_Classification_0"></a>Siemens Healthineers -  ECG Based Classification</h1>
<h3 class="code-line" data-line-start=2 data-line-end=3 ><a id="Challenge__2"></a>Challenge ⛓</h3>
<p class="has-line-data" data-line-start="4" data-line-end="5">To propose and implement an intelligent real time heartbeat classification algorithm and supplement results with Explainable AI</p>
<h3 class="code-line" data-line-start=6 data-line-end=7 ><a id="Description__6"></a>Description 🤖</h3>
<p class="has-line-data" data-line-start="7" data-line-end="8">Cardio Vascular Diseases happens to be the major contributor of death rate. Heartbeat is a basic physiological function of the human body and it indicates and helps a lot in investigation of heart function. One non-invasive method of assessing heart function is using an ECG. The dataset provided for this challenge has 17 classes of ECGs. There are attempts made classifying this data using different approaches, please refer online for sources. There is one article which is provided in the references section for you to understand the problem better</p>
<h3 class="code-line" data-line-start=9 data-line-end=10 ><a id="What_it_does__9"></a>What it does? 🪄</h3>
<p class="has-line-data" data-line-start="11" data-line-end="12">I devised an innovative algorithm, for the classification of ECG into 17 classes.</p>
<ul>
<li class="has-line-data" data-line-start="12" data-line-end="13">Firstly, the algorithm enhances the provided dataset, by using a roll-over technique, such that each class is populated with new cases and a balanced dataset is formed.</li>
<li class="has-line-data" data-line-start="13" data-line-end="15">Secondly, a dual path Deep Architecture is devised, with analysing various provided parameters.</li>
</ul>
<h3 class="code-line" data-line-start=15 data-line-end=16 ><a id="Devised_Algorithms__15"></a>Devised Algorithms 🔭</h3>
<ul>
<li class="has-line-data" data-line-start="17" data-line-end="22">
<h4 class="code-line" data-line-start=17 data-line-end=18 ><a id="RollOver_Technique__17"></a>Roll-Over Technique 🔬</h4>
<ul>
<li class="has-line-data" data-line-start="18" data-line-end="19">For the cases which have lestt cases in the provided dataset, the dataset for that class is equally rolled into multiple samples alternatively, using <code>numpy.roll()</code>.</li>
<li class="has-line-data" data-line-start="19" data-line-end="20">Here I used this rollover in clockwise and anticlockwise fashion on alternate samples from the dataset, to make the provided dataset rich.</li>
<li class="has-line-data" data-line-start="20" data-line-end="21">To balance the roll for each set, i rolled the dataset to a multiple of two, into clockwise and anticlockwise direction alternatively.</li>
<li class="has-line-data" data-line-start="21" data-line-end="22"><strong>Hence the transformed dataset is balanced, without losing its significance.</strong></li>
</ul>
</li>
</ul>
<hr>
<h4 class="code-line" data-line-start=23 data-line-end=24 ><a id="Dataset_is_transformed_to_following__23"></a>Dataset is transformed to following :</h4>
<table class="table table-striped table-bordered">
<thead>
<tr>
<th>ClassNum</th>
<th>ClassName</th>
<th>SetCount</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>6 WPW</td>
<td>273</td>
</tr>
<tr>
<td>1</td>
<td>5 SVTA</td>
<td>273</td>
</tr>
<tr>
<td>2</td>
<td>2 APB</td>
<td>264</td>
</tr>
<tr>
<td>3</td>
<td>15 RBBBB</td>
<td>248</td>
</tr>
<tr>
<td>4</td>
<td>11 IVR</td>
<td>280</td>
</tr>
<tr>
<td>5</td>
<td>4 AFIB</td>
<td>270</td>
</tr>
<tr>
<td>6</td>
<td>7 PVC</td>
<td>270</td>
</tr>
<tr>
<td>7</td>
<td>1 NSR</td>
<td>283</td>
</tr>
<tr>
<td>8</td>
<td>13 Fusion</td>
<td>275</td>
</tr>
<tr>
<td>9</td>
<td>9 Trigemy</td>
<td>273</td>
</tr>
<tr>
<td>10</td>
<td>3 AFL</td>
<td>280</td>
</tr>
<tr>
<td>11</td>
<td>12 VFL</td>
<td>280</td>
</tr>
<tr>
<td>12</td>
<td>14 LBBBB</td>
<td>206</td>
</tr>
<tr>
<td>13</td>
<td>16 SDHB</td>
<td>280</td>
</tr>
<tr>
<td>14</td>
<td>8 Bigeminy</td>
<td>275</td>
</tr>
<tr>
<td>15</td>
<td>17 PR</td>
<td>270</td>
</tr>
<tr>
<td>16</td>
<td>10 VT</td>
<td>280</td>
</tr>
</tbody>
</table>
<hr>
<ul>
<li class="has-line-data" data-line-start="45" data-line-end="56">
<h4 class="code-line" data-line-start=45 data-line-end=46 ><a id="Deep_Neural_Network_Architecture__45"></a>Deep Neural Network Architecture 🔬</h4>
<ul>
<li class="has-line-data" data-line-start="46" data-line-end="47">As I mentioned, the data has been transformed by using roll-over technique, so to counter it model has been made specifically such that, for one path it takes the given data, and for the other it takes the reverse of the same data. So it counters, the clockwise and anti-clockwise roll-over applied and gives accurate predictions.</li>
<li class="has-line-data" data-line-start="47" data-line-end="56"><strong>Model contains the following sub-models:</strong>
<ul>
<li class="has-line-data" data-line-start="48" data-line-end="50"><strong>Conv Block</strong>
<ul>
<li class="has-line-data" data-line-start="49" data-line-end="50"><img src="./assets/ConvBlock.jpg" alt="Conv Block"></li>
</ul>
</li>
<li class="has-line-data" data-line-start="50" data-line-end="51">Linear Block</li>
<li class="has-line-data" data-line-start="51" data-line-end="52">Attention Model</li>
<li class="has-line-data" data-line-start="52" data-line-end="53">InitConv Block</li>
<li class="has-line-data" data-line-start="53" data-line-end="54">Long-Short Term Memory Block</li>
<li class="has-line-data" data-line-start="54" data-line-end="56"></li>
</ul>
</li>
</ul>
</li>
</ul>