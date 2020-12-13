<p class="has-line-data" data-line-start="8" data-line-end="13">Cardio Vascular Diseases happens to be the major contributor of death rate. Heartbeat is a basic physiological function of the<br>
human body and it indicates and helps a lot in investigation of heart function. One non-invasive method of assessing heart<br>
function is using an ECG. The dataset provided for this challenge has 17 classes of ECGs. There are attempts made classifying this<br>
data using different approaches, please refer online for sources. There is one article which is provided in the references section for<br>
you to understand the problem better</p>
<h3 class="code-line" data-line-start=14 data-line-end=15 ><a id="What_it_does_14"></a>What it does?</h3>
<p class="has-line-data" data-line-start="16" data-line-end="19">I devised an innovative algorithm, for the classification of ECG into 17 classes.<br>
Firstly, the algorithm enhances the provided dataset, by using a roll-over technique, such that each class is populated with new cases and a balanced dataset is formed.<br>
Secondly, a dual path Deep Architecture is devised, with analysing various provided parameters.</p>
<ul>
<li class="has-line-data" data-line-start="19" data-line-end="23">Devised Algorithm
<ul>
<li class="has-line-data" data-line-start="20" data-line-end="23">Roll-Over Technique
<ul>
<li class="has-line-data" data-line-start="21" data-line-end="23">Dataset is analysed for the classes which have less test cases.</li>
</ul>
</li>
</ul>
</li>
</ul>
<h5 class="code-line" data-line-start=23 data-line-end=24 ><a id="Dataset_is_transformed_to_following_23"></a>Dataset is transformed to following:</h5>
<p class="has-line-data" data-line-start="24" data-line-end="42">|   | ClassName |TrainCaseCount |<br>
0                  6 WPW               273<br>
1                273<br>
2  264<br>
3  248<br>
4  280<br>
5 4 AFIB 270<br>
6 7 PVC 266<br>
7 1 NSR 283<br>
8 13 Fusion 275<br>
9 9 Trigeminy 273<br>
10 3 AFL 280<br>
11 12 VFL 280<br>
12 14 LBBBB 206<br>
13 16 SDHB 280<br>
14 8 Bigeminy 275<br>
15 17 PR 270<br>
16 10 VT 280</p>
<table class="table table-striped table-bordered">
<thead>
<tr>
<th>ClassNum</th>
<th>ClassName</th>
<th>TrainSetCount</th>
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
<td></td>
</tr>
<tr>
<td>4</td>
<td>11 IVR</td>
<td></td>
</tr>
<tr>
<td>5</td>
<td></td>
<td></td>
</tr>
<tr>
<td>6</td>
<td></td>
<td></td>
</tr>
<tr>
<td>7</td>
<td></td>
<td></td>
</tr>
<tr>
<td>8</td>
<td></td>
<td></td>
</tr>
<tr>
<td>9</td>
<td></td>
<td></td>
</tr>
<tr>
<td>10</td>
<td></td>
<td></td>
</tr>
<tr>
<td>11</td>
<td></td>
<td></td>
</tr>
<tr>
<td>12</td>
<td></td>
<td></td>
</tr>
<tr>
<td>13</td>
<td></td>
<td></td>
</tr>
<tr>
<td>14</td>
<td></td>
<td></td>
</tr>
<tr>
<td>15</td>
<td></td>
<td></td>
</tr>
<tr>
<td>16</td>
<td></td>
<td></td>
</tr>
</tbody>
</table>
