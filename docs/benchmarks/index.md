# Benchmarks

ordito runs on the GPU, and the libraries most mesh-processing code reaches for do not. This page
compares the two on the same inputs: ordito on CUDA against eleven reference libraries — trimesh,
libigl, Open3D, SciPy, NumPy, potpourri3d (geometry-central), MeshLab (through PyMeshLab), PyVista
(VTK), MeshLib, PyMeshFix and PyTorch3D, the one other library with GPU kernels.

Every comparison times the same quantity on both sides, and the correctness suite checks that the
two answers agree. A time is the median of repeated calls from Python, including the wait for the
GPU to finish, so it is what a caller measures. The inputs are real 3D scans from the Stanford
repository (from 16 thousand to 28 million triangles) and synthetic meshes that each stress one
cost driver: many components, long boundaries, needle triangles, high valence.

<!-- bench:provenance -->
*Measured on GPU: NVIDIA GeForce RTX 5090 · CPU: Intel Xeon w7-2495X*
<!-- /bench:provenance -->

## Summary

How long each library takes relative to ordito, over every function both implement: ordito is 1x,
and a library at 30x takes thirty times as long. Each function counts once: its ratio is first
averaged over the inputs it was measured on, then the functions are averaged. Both averages are
geometric means, the usual way to combine ratios: a 10x win and a 10x loss cancel out.

<!-- bench:summary -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="262" viewBox="0 0 640 262" role="img" aria-label="Time of each library relative to ordito"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="244"/><text class="bench-tick" x="196.0" y="257" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="244"/><text class="bench-tick" x="274.0" y="257" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="244"/><text class="bench-tick" x="352.0" y="257" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="244"/><text class="bench-tick" x="430.0" y="257" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="244"/><text class="bench-tick" x="508.0" y="257" text-anchor="middle">1000x</text><g><title>ordito (CUDA): the reference, 1x</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x</text></g><g><title>MeshLib: 11x ordito&#x27;s time, geometric mean over 118 functions; slower than ordito in 102</title><rect class="bench-reference" x="196" y="29.0" width="158.7" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="359.7" y="38.0">11x  (118)</text></g><g><title>PyTorch3D (CUDA): 11x ordito&#x27;s time, geometric mean over 36 functions; slower than ordito in 34</title><rect class="bench-reference" x="196" y="49.0" width="160.0" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="45.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="361.0" y="58.0">11x  (36)</text></g><g><title>NumPy: 13x ordito&#x27;s time, geometric mean over 35 functions; slower than ordito in 30</title><rect class="bench-reference" x="196" y="69.0" width="165.3" height="10"/><image class="bench-logo bench-logo-numpy" href="../assets/logos/numpy.png" x="172.6" y="65.0" width="16.9" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">NumPy</text><text class="bench-value" x="366.3" y="78.0">13x  (35)</text></g><g><title>Open3D: 22x ordito&#x27;s time, geometric mean over 71 functions; slower than ordito in 67</title><rect class="bench-reference" x="196" y="89.0" width="182.6" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="86.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="98.0" text-anchor="end">Open3D</text><text class="bench-value" x="383.6" y="98.0">22x  (71)</text></g><g><title>SciPy: 42x ordito&#x27;s time, geometric mean over 24 functions; slower than ordito in 24</title><rect class="bench-reference" x="196" y="109.0" width="204.6" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="105.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="118.0" text-anchor="end">SciPy</text><text class="bench-value" x="405.6" y="118.0">42x  (24)</text></g><g><title>potpourri3d: 66x ordito&#x27;s time, geometric mean over 20 functions; slower than ordito in 20</title><rect class="bench-reference" x="196" y="129.0" width="219.7" height="10"/><text class="bench-label" x="166" y="138.0" text-anchor="end">potpourri3d</text><text class="bench-value" x="420.7" y="138.0">66x  (20)</text></g><g><title>libigl: 93x ordito&#x27;s time, geometric mean over 95 functions; slower than ordito in 94</title><rect class="bench-reference" x="196" y="149.0" width="231.4" height="10"/><text class="bench-label" x="166" y="158.0" text-anchor="end">libigl</text><text class="bench-value" x="432.4" y="158.0">93x  (95)</text></g><g><title>MeshLab: 97x ordito&#x27;s time, geometric mean over 79 functions; slower than ordito in 79</title><rect class="bench-reference" x="196" y="169.0" width="233.0" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="165.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="178.0" text-anchor="end">MeshLab</text><text class="bench-value" x="434.0" y="178.0">97x  (79)</text></g><g><title>trimesh: 105x ordito&#x27;s time, geometric mean over 119 functions; slower than ordito in 115</title><rect class="bench-reference" x="196" y="189.0" width="235.5" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="190.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="198.0" text-anchor="end">trimesh</text><text class="bench-value" x="436.5" y="198.0">105x  (119)</text></g><g><title>PyVista: 113x ordito&#x27;s time, geometric mean over 64 functions; slower than ordito in 62</title><rect class="bench-reference" x="196" y="209.0" width="238.2" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="211.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="218.0" text-anchor="end">PyVista</text><text class="bench-value" x="439.2" y="218.0">113x  (64)</text></g><g><title>PyMeshFix: 502x ordito&#x27;s time, geometric mean over 4 functions; slower than ordito in 4</title><rect class="bench-reference" x="196" y="229.0" width="288.7" height="10"/><text class="bench-label" x="166" y="238.0" text-anchor="end">PyMeshFix</text><text class="bench-value" x="489.7" y="238.0">502x  (4)</text></g></svg>
<figcaption>Time relative to ordito (CUDA), geometric mean over every function both implement; in parentheses, the number of functions. Log scale. Runs stopped at the time limit are left out.</figcaption>
</figure>

??? note "Values"

    | Library | Time relative to ordito | Functions compared | ordito faster in |
    |---|---|---|---|
    | MeshLib | 11x | 118 | 102 |
    | PyTorch3D (CUDA) | 11x | 36 | 34 |
    | NumPy | 13x | 35 | 30 |
    | Open3D | 22x | 71 | 67 |
    | SciPy | 42x | 24 | 24 |
    | potpourri3d | 66x | 20 | 20 |
    | libigl | 93x | 95 | 94 |
    | MeshLab | 97x | 79 | 79 |
    | trimesh | 105x | 119 | 115 |
    | PyVista | 113x | 64 | 62 |
    | PyMeshFix | 502x | 4 | 4 |
<!-- /bench:summary -->

Libraries are compared on different sets of functions, so each bar measures one library against
ordito, not the libraries against each other. The libraries closest to ordito are the ones whose
functions are mostly fast, compiled code on small inputs; the gaps grow with the input.

## By module

ordito's speedup per module, against the fastest library measured on each input — the toughest
comparison available for each function. Each module has its own page, with a chart for every
function.

<!-- bench:modules -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="922" viewBox="0 0 640 922" role="img" aria-label="ordito&#x27;s speedup per module"><line class="bench-baseline" x1="196.0" y1="0" x2="196.0" y2="904"/><text class="bench-tick" x="196.0" y="917" text-anchor="middle">1x</text><line class="bench-grid" x1="300.0" y1="0" x2="300.0" y2="904"/><text class="bench-tick" x="300.0" y="917" text-anchor="middle">10x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="904"/><text class="bench-tick" x="404.0" y="917" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="904"/><text class="bench-tick" x="508.0" y="917" text-anchor="middle">1000x</text><g><title>adjacency: ordito 189x faster than the fastest library, geometric mean over 7 functions; faster in 7</title><rect class="bench-subject" x="196" y="9.0" width="236.7" height="10"/><text class="bench-label" x="190" y="18.0" text-anchor="end">adjacency</text><text class="bench-value" x="437.7" y="18.0">189x  (7)</text></g><g><title>triangles: ordito 173x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="29.0" width="232.9" height="10"/><text class="bench-label" x="190" y="38.0" text-anchor="end">triangles</text><text class="bench-value" x="433.9" y="38.0">173x  (9)</text></g><g><title>tangent_space: ordito 153x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="49.0" width="227.3" height="10"/><text class="bench-label" x="190" y="58.0" text-anchor="end">tangent_space</text><text class="bench-value" x="428.3" y="58.0">153x  (2)</text></g><g><title>geodesic_walk: ordito 116x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="69.0" width="214.6" height="10"/><text class="bench-label" x="190" y="78.0" text-anchor="end">geodesic_walk</text><text class="bench-value" x="415.6" y="78.0">116x  (5)</text></g><g><title>edges: ordito 93x faster than the fastest library, geometric mean over 14 functions; faster in 14</title><rect class="bench-subject" x="196" y="89.0" width="204.9" height="10"/><text class="bench-label" x="190" y="98.0" text-anchor="end">edges</text><text class="bench-value" x="405.9" y="98.0">93x  (14)</text></g><g><title>graph: ordito 91x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="109.0" width="203.6" height="10"/><text class="bench-label" x="190" y="118.0" text-anchor="end">graph</text><text class="bench-value" x="404.6" y="118.0">91x  (6)</text></g><g><title>grouping: ordito 86x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="129.0" width="201.0" height="10"/><text class="bench-label" x="190" y="138.0" text-anchor="end">grouping</text><text class="bench-value" x="402.0" y="138.0">86x  (4)</text></g><g><title>validation: ordito 69x faster than the fastest library, geometric mean over 11 functions; faster in 11</title><rect class="bench-subject" x="196" y="149.0" width="191.0" height="10"/><text class="bench-label" x="190" y="158.0" text-anchor="end">validation</text><text class="bench-value" x="392.0" y="158.0">69x  (11)</text></g><g><title>smoothing: ordito 66x faster than the fastest library, geometric mean over 16 functions; faster in 15</title><rect class="bench-subject" x="196" y="169.0" width="189.3" height="10"/><text class="bench-label" x="190" y="178.0" text-anchor="end">smoothing</text><text class="bench-value" x="390.3" y="178.0">66x  (16)</text></g><g><title>seams: ordito 58x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="189.0" width="183.3" height="10"/><text class="bench-label" x="190" y="198.0" text-anchor="end">seams</text><text class="bench-value" x="384.3" y="198.0">58x  (4)</text></g><g><title>curvature: ordito 53x faster than the fastest library, geometric mean over 3 functions; faster in 3</title><rect class="bench-subject" x="196" y="209.0" width="179.2" height="10"/><text class="bench-label" x="190" y="218.0" text-anchor="end">curvature</text><text class="bench-value" x="380.2" y="218.0">53x  (3)</text></g><g><title>repair: ordito 47x faster than the fastest library, geometric mean over 19 functions; faster in 17</title><rect class="bench-subject" x="196" y="229.0" width="174.1" height="10"/><text class="bench-label" x="190" y="238.0" text-anchor="end">repair</text><text class="bench-value" x="375.1" y="238.0">47x  (19)</text></g><g><title>laplacian: ordito 40x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="249.0" width="166.6" height="10"/><text class="bench-label" x="190" y="258.0" text-anchor="end">laplacian</text><text class="bench-value" x="367.6" y="258.0">40x  (9)</text></g><g><title>visibility: ordito 35x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="269.0" width="160.7" height="10"/><text class="bench-label" x="190" y="278.0" text-anchor="end">visibility</text><text class="bench-value" x="361.7" y="278.0">35x  (6)</text></g><g><title>metrics: ordito 33x faster than the fastest library, geometric mean over 7 functions; faster in 7</title><rect class="bench-subject" x="196" y="289.0" width="157.5" height="10"/><text class="bench-label" x="190" y="298.0" text-anchor="end">metrics</text><text class="bench-value" x="358.5" y="298.0">33x  (7)</text></g><g><title>mesh: ordito 31x faster than the fastest library, geometric mean over 13 functions; faster in 13</title><rect class="bench-subject" x="196" y="309.0" width="155.6" height="10"/><text class="bench-label" x="190" y="318.0" text-anchor="end">mesh</text><text class="bench-value" x="356.6" y="318.0">31x  (13)</text></g><g><title>levelset: ordito 31x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="329.0" width="154.4" height="10"/><text class="bench-label" x="190" y="338.0" text-anchor="end">levelset</text><text class="bench-value" x="355.4" y="338.0">31x  (4)</text></g><g><title>energies: ordito 28x faster than the fastest library, geometric mean over 8 functions; faster in 8</title><rect class="bench-subject" x="196" y="349.0" width="150.8" height="10"/><text class="bench-label" x="190" y="358.0" text-anchor="end">energies</text><text class="bench-value" x="351.8" y="358.0">28x  (8)</text></g><g><title>measures: ordito 28x faster than the fastest library, geometric mean over 3 functions; faster in 3</title><rect class="bench-subject" x="196" y="369.0" width="150.2" height="10"/><text class="bench-label" x="190" y="378.0" text-anchor="end">measures</text><text class="bench-value" x="351.2" y="378.0">28x  (3)</text></g><g><title>interpolation: ordito 27x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="389.0" width="148.7" height="10"/><text class="bench-label" x="190" y="398.0" text-anchor="end">interpolation</text><text class="bench-value" x="349.7" y="398.0">27x  (5)</text></g><g><title>remesh: ordito 24x faster than the fastest library, geometric mean over 10 functions; faster in 10</title><rect class="bench-subject" x="196" y="409.0" width="143.9" height="10"/><text class="bench-label" x="190" y="418.0" text-anchor="end">remesh</text><text class="bench-value" x="344.9" y="418.0">24x  (10)</text></g><g><title>vertices: ordito 23x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="429.0" width="142.1" height="10"/><text class="bench-label" x="190" y="438.0" text-anchor="end">vertices</text><text class="bench-value" x="343.1" y="438.0">23x  (4)</text></g><g><title>points: ordito 21x faster than the fastest library, geometric mean over 19 functions; faster in 19</title><rect class="bench-subject" x="196" y="449.0" width="138.3" height="10"/><text class="bench-label" x="190" y="458.0" text-anchor="end">points</text><text class="bench-value" x="339.3" y="458.0">21x  (19)</text></g><g><title>creation: ordito 21x faster than the fastest library, geometric mean over 20 functions; faster in 16</title><rect class="bench-subject" x="196" y="469.0" width="137.6" height="10"/><text class="bench-label" x="190" y="478.0" text-anchor="end">creation</text><text class="bench-value" x="338.6" y="478.0">21x  (20)</text></g><g><title>neighbors: ordito 21x faster than the fastest library, geometric mean over 15 functions; faster in 15</title><rect class="bench-subject" x="196" y="489.0" width="136.6" height="10"/><text class="bench-label" x="190" y="498.0" text-anchor="end">neighbors</text><text class="bench-value" x="337.6" y="498.0">21x  (15)</text></g><g><title>holes: ordito 19x faster than the fastest library, geometric mean over 12 functions; faster in 12</title><rect class="bench-subject" x="196" y="509.0" width="133.8" height="10"/><text class="bench-label" x="190" y="518.0" text-anchor="end">holes</text><text class="bench-value" x="334.8" y="518.0">19x  (12)</text></g><g><title>intersection: ordito 19x faster than the fastest library, geometric mean over 11 functions; faster in 9</title><rect class="bench-subject" x="196" y="529.0" width="132.8" height="10"/><text class="bench-label" x="190" y="538.0" text-anchor="end">intersection</text><text class="bench-value" x="333.8" y="538.0">19x  (11)</text></g><g><title>array: ordito 17x faster than the fastest library, geometric mean over 19 functions; faster in 18</title><rect class="bench-subject" x="196" y="549.0" width="128.3" height="10"/><text class="bench-label" x="190" y="558.0" text-anchor="end">array</text><text class="bench-value" x="329.3" y="558.0">17x  (19)</text></g><g><title>heat: ordito 17x faster than the fastest library, geometric mean over 8 functions; faster in 8</title><rect class="bench-subject" x="196" y="569.0" width="127.1" height="10"/><text class="bench-label" x="190" y="578.0" text-anchor="end">heat</text><text class="bench-value" x="328.1" y="578.0">17x  (8)</text></g><g><title>sample: ordito 14x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="589.0" width="120.2" height="10"/><text class="bench-label" x="190" y="598.0" text-anchor="end">sample</text><text class="bench-value" x="321.2" y="598.0">14x  (2)</text></g><g><title>transform: ordito 12x faster than the fastest library, geometric mean over 5 functions; faster in 3</title><rect class="bench-subject" x="196" y="609.0" width="110.9" height="10"/><text class="bench-label" x="190" y="618.0" text-anchor="end">transform</text><text class="bench-value" x="311.9" y="618.0">12x  (5)</text></g><g><title>linalg: ordito 9.9x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="629.0" width="103.5" height="10"/><text class="bench-label" x="190" y="638.0" text-anchor="end">linalg</text><text class="bench-value" x="304.5" y="638.0">9.9x  (2)</text></g><g><title>parametrization: ordito 9.7x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="649.0" width="102.5" height="10"/><text class="bench-label" x="190" y="658.0" text-anchor="end">parametrization</text><text class="bench-value" x="303.5" y="658.0">9.7x  (6)</text></g><g><title>selection: ordito 9.2x faster than the fastest library, geometric mean over 8 functions; faster in 5</title><rect class="bench-subject" x="196" y="669.0" width="100.3" height="10"/><text class="bench-label" x="190" y="678.0" text-anchor="end">selection</text><text class="bench-value" x="301.3" y="678.0">9.2x  (8)</text></g><g><title>registration: ordito 8.3x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="689.0" width="95.4" height="10"/><text class="bench-label" x="190" y="698.0" text-anchor="end">registration</text><text class="bench-value" x="296.4" y="698.0">8.3x  (5)</text></g><g><title>ray: ordito 8.2x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="709.0" width="94.9" height="10"/><text class="bench-label" x="190" y="718.0" text-anchor="end">ray</text><text class="bench-value" x="295.9" y="718.0">8.2x  (5)</text></g><g><title>boundary: ordito 7.9x faster than the fastest library, geometric mean over 7 functions; faster in 5</title><rect class="bench-subject" x="196" y="729.0" width="93.3" height="10"/><text class="bench-label" x="190" y="738.0" text-anchor="end">boundary</text><text class="bench-value" x="294.3" y="738.0">7.9x  (7)</text></g><g><title>proximity: ordito 7.2x faster than the fastest library, geometric mean over 6 functions; faster in 5</title><rect class="bench-subject" x="196" y="749.0" width="88.9" height="10"/><text class="bench-label" x="190" y="758.0" text-anchor="end">proximity</text><text class="bench-value" x="289.9" y="758.0">7.2x  (6)</text></g><g><title>combine: ordito 7.1x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="769.0" width="88.3" height="10"/><text class="bench-label" x="190" y="778.0" text-anchor="end">combine</text><text class="bench-value" x="289.3" y="778.0">7.1x  (2)</text></g><g><title>voxels: ordito 6.4x faster than the fastest library, geometric mean over 15 functions; faster in 12</title><rect class="bench-subject" x="196" y="789.0" width="83.6" height="10"/><text class="bench-label" x="190" y="798.0" text-anchor="end">voxels</text><text class="bench-value" x="284.6" y="798.0">6.4x  (15)</text></g><g><title>reconstruction: ordito 6.1x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="809.0" width="82.0" height="10"/><text class="bench-label" x="190" y="818.0" text-anchor="end">reconstruction</text><text class="bench-value" x="283.0" y="818.0">6.1x  (6)</text></g><g><title>polyline: ordito 4.7x faster than the fastest library, geometric mean over 7 functions; faster in 5</title><rect class="bench-subject" x="196" y="829.0" width="70.0" height="10"/><text class="bench-label" x="190" y="838.0" text-anchor="end">polyline</text><text class="bench-value" x="271.0" y="838.0">4.7x  (7)</text></g><g><title>reduce: ordito 3.6x faster than the fastest library, geometric mean over 15 functions; faster in 10</title><rect class="bench-subject" x="196" y="849.0" width="57.6" height="10"/><text class="bench-label" x="190" y="858.0" text-anchor="end">reduce</text><text class="bench-value" x="258.6" y="858.0">3.6x  (15)</text></g><g><title>homology: ordito 3.2x faster than the fastest library, geometric mean over 1 functions; faster in 1</title><rect class="bench-subject" x="196" y="869.0" width="52.5" height="10"/><text class="bench-label" x="190" y="878.0" text-anchor="end">homology</text><text class="bench-value" x="253.5" y="878.0">3.2x  (1)</text></g><g><title>bounds: ordito 1.2x faster than the fastest library, geometric mean over 9 functions; faster in 7</title><rect class="bench-subject" x="196" y="889.0" width="8.9" height="10"/><text class="bench-label" x="190" y="898.0" text-anchor="end">bounds</text><text class="bench-value" x="209.9" y="898.0">1.2x  (9)</text></g></svg>
<figcaption>ordito (CUDA)'s speedup over the fastest library measured on each input, geometric mean per module; in parentheses, the number of functions. Log scale.</figcaption>
</figure>

??? note "Values"

    | Module | ordito's speedup | Functions | ordito faster in |
    |---|---|---|---|
    | [`adjacency`](adjacency.md) | 189x | 7 | 7 |
    | [`triangles`](triangles.md) | 173x | 9 | 9 |
    | [`tangent_space`](tangent_space.md) | 153x | 2 | 2 |
    | [`geodesic_walk`](geodesic_walk.md) | 116x | 5 | 5 |
    | [`edges`](edges.md) | 93x | 14 | 14 |
    | [`graph`](graph.md) | 91x | 6 | 6 |
    | [`grouping`](grouping.md) | 86x | 4 | 4 |
    | [`validation`](validation.md) | 69x | 11 | 11 |
    | [`smoothing`](smoothing.md) | 66x | 16 | 15 |
    | [`seams`](seams.md) | 58x | 4 | 4 |
    | [`curvature`](curvature.md) | 53x | 3 | 3 |
    | [`repair`](repair.md) | 47x | 19 | 17 |
    | [`laplacian`](laplacian.md) | 40x | 9 | 9 |
    | [`visibility`](visibility.md) | 35x | 6 | 6 |
    | [`metrics`](metrics.md) | 33x | 7 | 7 |
    | [`mesh`](mesh.md) | 31x | 13 | 13 |
    | [`levelset`](levelset.md) | 31x | 4 | 4 |
    | [`energies`](energies.md) | 28x | 8 | 8 |
    | [`measures`](measures.md) | 28x | 3 | 3 |
    | [`interpolation`](interpolation.md) | 27x | 5 | 5 |
    | [`remesh`](remesh.md) | 24x | 10 | 10 |
    | [`vertices`](vertices.md) | 23x | 4 | 4 |
    | [`points`](points.md) | 21x | 19 | 19 |
    | [`creation`](creation.md) | 21x | 20 | 16 |
    | [`neighbors`](neighbors.md) | 21x | 15 | 15 |
    | [`holes`](holes.md) | 19x | 12 | 12 |
    | [`intersection`](intersection.md) | 19x | 11 | 9 |
    | [`array`](array.md) | 17x | 19 | 18 |
    | [`heat`](heat.md) | 17x | 8 | 8 |
    | [`sample`](sample.md) | 14x | 2 | 2 |
    | [`transform`](transform.md) | 12x | 5 | 3 |
    | [`linalg`](linalg.md) | 9.9x | 2 | 2 |
    | [`parametrization`](parametrization.md) | 9.7x | 6 | 6 |
    | [`selection`](selection.md) | 9.2x | 8 | 5 |
    | [`registration`](registration.md) | 8.3x | 5 | 5 |
    | [`ray`](ray.md) | 8.2x | 5 | 5 |
    | [`boundary`](boundary.md) | 7.9x | 7 | 5 |
    | [`proximity`](proximity.md) | 7.2x | 6 | 5 |
    | [`combine`](combine.md) | 7.1x | 2 | 2 |
    | [`voxels`](voxels.md) | 6.4x | 15 | 12 |
    | [`reconstruction`](reconstruction.md) | 6.1x | 6 | 6 |
    | [`polyline`](polyline.md) | 4.7x | 7 | 5 |
    | [`reduce`](reduce.md) | 3.6x | 15 | 10 |
    | [`homology`](homology.md) | 3.2x | 1 | 1 |
    | [`bounds`](bounds.md) | 1.2x | 9 | 7 |
<!-- /bench:modules -->

## Highlights

The comparisons with the widest field of libraries. Each chart shows one input, with ordito at 1x
and every other library's time relative to it; each module's page has every function and every
input.

### Signed distance to a mesh

Signed distance from a point set to a triangle mesh, the widest field among the spatial queries.

<!-- bench:cell test_proximity/signed_distance_on_mesh -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="signed_distance_on_mesh on dragon (871 414 faces) · sign_mode=parity"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="124"/><text class="bench-tick" x="258.4" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="124"/><text class="bench-tick" x="320.8" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="124"/><text class="bench-tick" x="383.2" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="124"/><text class="bench-tick" x="445.6" y="137" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 5.09 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 5.09 ms</text></g><g><title>MeshLib: 94.85 ms, 19x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="141.7" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="342.7" y="38.0">19x · 94.85 ms</text></g><g><title>Open3D: 281.78 ms, 55x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="171.2" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="372.2" y="58.0">55x · 281.78 ms</text></g><g><title>PyVista: 1.024 s, 201x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="206.1" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="407.1" y="78.0">201x · 1.024 s</text></g><g><title>libigl: 1.045 s, 205x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="206.7" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="407.7" y="98.0">205x · 1.045 s</text></g><g><title>MeshLab: 7.216 s, 1417x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="259.0" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="105.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">MeshLab</text><text class="bench-value" x="460.0" y="118.0">1417x · 7.216 s</text></g></svg>
<figcaption>dragon (871 414 faces) · sign_mode=parity · time relative to ordito, log scale · [all proximity benchmarks](proximity.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 5.09 ms | 1x |
    | MeshLib | 94.85 ms | 19x |
    | Open3D | 281.78 ms | 55x |
    | PyVista | 1.024 s | 201x |
    | libigl | 1.045 s | 205x |
    | MeshLab | 7.216 s | 1417x |
<!-- /bench:cell test_proximity/signed_distance_on_mesh -->

### Nearest neighbours, k = 7

Seven nearest neighbours of every query point, against CPU trees and PyTorch3D's GPU kernel.

<!-- bench:cell test_neighbors/query_nearest_bvh_k7 -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="query_nearest_bvh_k7 on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="104"/><text class="bench-tick" x="300.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="104"/><text class="bench-tick" x="404.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">100x</text><g><title>ordito (CUDA): 1.73 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 1.73 ms</text></g><g><title>PyTorch3D (CUDA): 43.89 ms, 25x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="249.9" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="25.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="450.9" y="38.0">25x · 43.89 ms</text></g><g><title>Open3D: 117.91 ms, 68x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="294.6" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="495.6" y="58.0">68x · 117.91 ms</text></g><g><title>SciPy: 147.45 ms, 85x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="304.7" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="65.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="78.0" text-anchor="end">SciPy</text><text class="bench-value" x="505.7" y="78.0">85x · 147.45 ms</text></g><g><title>libigl: out of memory</title><path class="bench-failed" d="M198,90.0L206,98.0M206,90.0L198,98.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="212.0" y="98.0">out of memory</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · a red cross is a run that left no time · [all neighbors benchmarks](neighbors.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.73 ms | 1x |
    | PyTorch3D (CUDA) | 43.89 ms | 25x |
    | Open3D | 117.91 ms | 68x |
    | SciPy | 147.45 ms | 85x |
    | libigl | out of memory | |
<!-- /bench:cell test_neighbors/query_nearest_bvh_k7 -->

### Nearest neighbour, k = 1

The same query with a single neighbour. Every library with a tree, ordito included, builds it
over the cloud inside the timed call, and the build is most of the work: about two thirds of
ordito's time.

<!-- bench:cell test_neighbors/query_nearest_bvh_k1 -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="query_nearest_bvh_k1 on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="124"/><text class="bench-tick" x="274.0" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="124"/><text class="bench-tick" x="352.0" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="124"/><text class="bench-tick" x="430.0" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 705.4 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 705.4 µs</text></g><g><title>MeshLib: 16.05 ms, 23x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="183.8" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="384.8" y="38.0">23x · 16.05 ms</text></g><g><title>PyTorch3D (CUDA): 33.73 ms, 48x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="209.0" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="45.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="410.0" y="58.0">48x · 33.73 ms</text></g><g><title>Open3D: 118.98 ms, 169x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="251.7" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="452.7" y="78.0">169x · 118.98 ms</text></g><g><title>SciPy: 135.47 ms, 192x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="256.1" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="85.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="98.0" text-anchor="end">SciPy</text><text class="bench-value" x="457.1" y="98.0">192x · 135.47 ms</text></g><g><title>libigl: out of memory</title><path class="bench-failed" d="M198,110.0L206,118.0M206,110.0L198,118.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">libigl</text><text class="bench-value" x="212.0" y="118.0">out of memory</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · a red cross is a run that left no time · [all neighbors benchmarks](neighbors.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 705.4 µs | 1x |
    | MeshLib | 16.05 ms | 23x |
    | PyTorch3D (CUDA) | 33.73 ms | 48x |
    | Open3D | 118.98 ms | 169x |
    | SciPy | 135.47 ms | 192x |
    | libigl | out of memory | |
<!-- /bench:cell test_neighbors/query_nearest_bvh_k1 -->

### Quadric decimation

Edge-collapse simplification under the quadric error metric, against five independent
decimators.

<!-- bench:cell test_remesh/quadric_decimate -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="quadric_decimate on saddle (34 848 faces) · target_ratio=0.5"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="124"/><text class="bench-tick" x="300.0" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="124"/><text class="bench-tick" x="404.0" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">100x</text><g><title>ordito (CUDA): 3.10 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 3.10 ms</text></g><g><title>MeshLib: 19.08 ms, 6.2x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="186.1" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="387.1" y="38.0">6.2x · 19.08 ms</text></g><g><title>Open3D: 45.93 ms, 15x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="225.7" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="426.7" y="58.0">15x · 45.93 ms</text></g><g><title>PyVista: 46.34 ms, 15x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="226.1" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="427.1" y="78.0">15x · 46.34 ms</text></g><g><title>libigl: 51.05 ms, 16x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="230.5" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="431.5" y="98.0">16x · 51.05 ms</text></g><g><title>MeshLab: 177.33 ms, 57x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="286.8" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="105.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">MeshLab</text><text class="bench-value" x="487.8" y="118.0">57x · 177.33 ms</text></g></svg>
<figcaption>saddle (34 848 faces) · target_ratio=0.5 · time relative to ordito, log scale · [all remesh benchmarks](remesh.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 3.10 ms | 1x |
    | MeshLib | 19.08 ms | 6.2x |
    | Open3D | 45.93 ms | 15x |
    | PyVista | 46.34 ms | 15x |
    | libigl | 51.05 ms | 16x |
    | MeshLab | 177.33 ms | 57x |
<!-- /bench:cell test_remesh/quadric_decimate -->

### Minimal oriented bounding box

Every library here searches for the minimum-volume box, not the PCA box, so the bars compare
the same optimisation.

<!-- bench:cell test_bounds/oriented_bounding_box -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="oriented_bounding_box on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 1.50 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 1.50 ms</text></g><g><title>PyVista: 18.34 ms, 12x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="130.3" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="331.3" y="38.0">12x · 18.34 ms</text></g><g><title>trimesh: 159.09 ms, 106x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="188.8" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="389.8" y="58.0">106x · 159.09 ms</text></g><g><title>Open3D: 2.096 s, 1400x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="258.7" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="459.7" y="78.0">1400x · 2.096 s</text></g><g><title>libigl: 2.483 s, 1659x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="263.3" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="464.3" y="98.0">1659x · 2.483 s</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all bounds benchmarks](bounds.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.50 ms | 1x |
    | PyVista | 18.34 ms | 12x |
    | trimesh | 159.09 ms | 106x |
    | Open3D | 2.096 s | 1400x |
    | libigl | 2.483 s | 1659x |
<!-- /bench:cell test_bounds/oriented_bounding_box -->

### Convex hull

<!-- bench:cell test_points/convex_hull -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="convex_hull on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 2.02 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 2.02 ms</text></g><g><title>Open3D: 49.37 ms, 24x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="186.3" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="26.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="38.0" text-anchor="end">Open3D</text><text class="bench-value" x="387.3" y="38.0">24x · 49.37 ms</text></g><g><title>SciPy: 61.31 ms, 30x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="193.7" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="45.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="58.0" text-anchor="end">SciPy</text><text class="bench-value" x="394.7" y="58.0">30x · 61.31 ms</text></g><g><title>trimesh: 79.55 ms, 39x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="202.5" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="70.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="78.0" text-anchor="end">trimesh</text><text class="bench-value" x="403.5" y="78.0">39x · 79.55 ms</text></g><g><title>MeshLab: 417.44 ms, 207x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="258.6" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="459.6" y="98.0">207x · 417.44 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all points benchmarks](points.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 2.02 ms | 1x |
    | Open3D | 49.37 ms | 24x |
    | SciPy | 61.31 ms | 30x |
    | trimesh | 79.55 ms | 39x |
    | MeshLab | 417.44 ms | 207x |
<!-- /bench:cell test_points/convex_hull -->

### Blue-noise sampling

Poisson-disk sampling of a surface: four different algorithms for the same distribution.

<!-- bench:cell test_sample/blue_noise -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="blue_noise on dragon (871 414 faces) · radius_scale=1.0"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 1.50 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 1.50 ms</text></g><g><title>MeshLib: 12.08 ms, 8.1x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="148.8" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="349.8" y="38.0">8.1x · 12.08 ms</text></g><g><title>Open3D: 57.02 ms, 38x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="201.3" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="402.3" y="58.0">38x · 57.02 ms</text></g><g><title>libigl: 304.39 ms, 204x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="258.1" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="459.1" y="78.0">204x · 304.39 ms</text></g><g><title>MeshLab: 488.56 ms, 327x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="274.1" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="475.1" y="98.0">327x · 488.56 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · radius_scale=1.0 · time relative to ordito, log scale · [all sample benchmarks](sample.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.50 ms | 1x |
    | MeshLib | 12.08 ms | 8.1x |
    | Open3D | 57.02 ms | 38x |
    | libigl | 304.39 ms | 204x |
    | MeshLab | 488.56 ms | 327x |
<!-- /bench:cell test_sample/blue_noise -->

### Geodesic distance

Geodesic distance from a source vertex by the heat method, including building the solver.

<!-- bench:cell test_heat/heat_geodesic -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="heat_geodesic on sphere_med (81 920 faces) · setup=full"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="104"/><text class="bench-tick" x="300.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="104"/><text class="bench-tick" x="404.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">100x</text><g><title>ordito (CUDA): 6.45 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 6.45 ms</text></g><g><title>PyVista: 23.72 ms, 3.7x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="162.8" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="363.8" y="38.0">3.7x · 23.72 ms</text></g><g><title>potpourri3d: 539.58 ms, 84x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="303.9" height="10"/><text class="bench-label" x="166" y="58.0" text-anchor="end">potpourri3d</text><text class="bench-value" x="504.9" y="58.0">84x · 539.58 ms</text></g><g><title>libigl: 597.39 ms, 93x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="308.5" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="509.5" y="78.0">93x · 597.39 ms</text></g><g><title>MeshLab: 601.38 ms, 93x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="308.8" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="509.8" y="98.0">93x · 601.38 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · setup=full · time relative to ordito, log scale · [all heat benchmarks](heat.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 6.45 ms | 1x |
    | PyVista | 23.72 ms | 3.7x |
    | potpourri3d | 539.58 ms | 84x |
    | libigl | 597.39 ms | 93x |
    | MeshLab | 601.38 ms | 93x |
<!-- /bench:cell test_heat/heat_geodesic -->

### Repair: merging duplicate vertices

<!-- bench:cell test_repair/remove_duplicated_vertices -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="remove_duplicated_vertices on sphere_med (81 920 faces) · epsilon=0.0"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 234.1 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 234.1 µs</text></g><g><title>Open3D: 16.42 ms, 70x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="222.0" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="26.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="38.0" text-anchor="end">Open3D</text><text class="bench-value" x="423.0" y="38.0">70x · 16.42 ms</text></g><g><title>PyVista: 25.66 ms, 110x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="237.1" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="51.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyVista</text><text class="bench-value" x="438.1" y="58.0">110x · 25.66 ms</text></g><g><title>MeshLib: 31.33 ms, 134x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="243.9" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="67.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">MeshLib</text><text class="bench-value" x="444.9" y="78.0">134x · 31.33 ms</text></g><g><title>MeshLab: 221.69 ms, 947x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="310.2" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="511.2" y="98.0">947x · 221.69 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · epsilon=0.0 · time relative to ordito, log scale · [all repair benchmarks](repair.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 234.1 µs | 1x |
    | Open3D | 16.42 ms | 70x |
    | PyVista | 25.66 ms | 110x |
    | MeshLib | 31.33 ms | 134x |
    | MeshLab | 221.69 ms | 947x |
<!-- /bench:cell test_repair/remove_duplicated_vertices -->

### Repair: consistent winding

Re-orienting every face to agree with its neighbours, a whole-mesh propagation.

<!-- bench:cell test_repair/make_winding_consistent -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="make_winding_consistent on sphere_med (81 920 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 128.7 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 128.7 µs</text></g><g><title>MeshLib: 6.91 ms, 54x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="212.9" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="413.9" y="38.0">54x · 6.91 ms</text></g><g><title>trimesh: 17.97 ms, 140x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="245.3" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="446.3" y="58.0">140x · 17.97 ms</text></g><g><title>libigl: 36.63 ms, 285x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="269.4" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="470.4" y="78.0">285x · 36.63 ms</text></g><g><title>MeshLab: 51.21 ms, 398x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="280.8" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="481.8" y="98.0">398x · 51.21 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · time relative to ordito, log scale · [all repair benchmarks](repair.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 128.7 µs | 1x |
    | MeshLib | 6.91 ms | 54x |
    | trimesh | 17.97 ms | 140x |
    | libigl | 36.63 ms | 285x |
    | MeshLab | 51.21 ms | 398x |
<!-- /bench:cell test_repair/make_winding_consistent -->

### Connected components

<!-- bench:cell test_adjacency/face_connected_component_labels -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="102" viewBox="0 0 640 102" role="img" aria-label="face_connected_component_labels on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="84"/><text class="bench-tick" x="196.0" y="97" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="84"/><text class="bench-tick" x="274.0" y="97" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="84"/><text class="bench-tick" x="352.0" y="97" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="84"/><text class="bench-tick" x="430.0" y="97" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="84"/><text class="bench-tick" x="508.0" y="97" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 553.1 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 553.1 µs</text></g><g><title>MeshLib: 21.47 ms, 39x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="202.0" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="403.0" y="38.0">39x · 21.47 ms</text></g><g><title>PyVista: 280.04 ms, 506x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="288.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="51.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyVista</text><text class="bench-value" x="489.9" y="58.0">506x · 280.04 ms</text></g><g><title>libigl: 447.65 ms, 809x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="304.8" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="505.8" y="78.0">809x · 447.65 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all adjacency benchmarks](adjacency.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 553.1 µs | 1x |
    | MeshLib | 21.47 ms | 39x |
    | PyVista | 280.04 ms | 506x |
    | libigl | 447.65 ms | 809x |
<!-- /bench:cell test_adjacency/face_connected_component_labels -->

### Unique edges

The undirected edge list, the first step of most topology code.

<!-- bench:cell test_edges/edges_unique -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="edges_unique on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 457.6 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 457.6 µs</text></g><g><title>PyTorch3D (CUDA): 3.52 ms, 7.7x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="147.1" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="25.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="348.1" y="38.0">7.7x · 3.52 ms</text></g><g><title>trimesh: 243.05 ms, 531x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="290.6" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="491.6" y="58.0">531x · 243.05 ms</text></g><g><title>PyVista: 268.58 ms, 587x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="293.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="494.9" y="78.0">587x · 268.58 ms</text></g><g><title>libigl: 330.25 ms, 722x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="300.9" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="501.9" y="98.0">722x · 330.25 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all edges benchmarks](edges.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 457.6 µs | 1x |
    | PyTorch3D (CUDA) | 3.52 ms | 7.7x |
    | trimesh | 243.05 ms | 531x |
    | PyVista | 268.58 ms | 587x |
    | libigl | 330.25 ms | 722x |
<!-- /bench:cell test_edges/edges_unique -->

### Taubin smoothing

<!-- bench:cell test_smoothing/filter_taubin -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="filter_taubin on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 573.5 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 573.5 µs</text></g><g><title>PyVista: 153.29 ms, 267x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="213.8" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="414.8" y="38.0">267x · 153.29 ms</text></g><g><title>trimesh: 198.54 ms, 346x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="220.9" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="421.9" y="58.0">346x · 198.54 ms</text></g><g><title>Open3D: 306.02 ms, 534x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="232.6" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="433.6" y="78.0">534x · 306.02 ms</text></g><g><title>MeshLab: 867.52 ms, 1513x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="260.8" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="461.8" y="98.0">1513x · 867.52 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all smoothing benchmarks](smoothing.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 573.5 µs | 1x |
    | PyVista | 153.29 ms | 267x |
    | trimesh | 198.54 ms | 346x |
    | Open3D | 306.02 ms | 534x |
    | MeshLab | 867.52 ms | 1513x |
<!-- /bench:cell test_smoothing/filter_taubin -->

### Voxelizing a mesh

<!-- bench:cell test_voxels/voxelize_mesh -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="voxelize_mesh on dragon (871 414 faces) · divisor=64"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 639.9 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 639.9 µs</text></g><g><title>MeshLib: 109.34 ms, 171x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="201.7" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="402.7" y="38.0">171x · 109.34 ms</text></g><g><title>Open3D: 161.89 ms, 253x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="212.4" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="413.4" y="58.0">253x · 161.89 ms</text></g><g><title>PyVista: 255.97 ms, 400x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="224.8" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="425.8" y="78.0">400x · 255.97 ms</text></g><g><title>trimesh: 655.79 ms, 1025x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="250.3" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="90.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="98.0" text-anchor="end">trimesh</text><text class="bench-value" x="451.3" y="98.0">1025x · 655.79 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · divisor=64 · time relative to ordito, log scale · [all voxels benchmarks](voxels.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 639.9 µs | 1x |
    | MeshLib | 109.34 ms | 171x |
    | Open3D | 161.89 ms | 253x |
    | PyVista | 255.97 ms | 400x |
    | trimesh | 655.79 ms | 1025x |
<!-- /bench:cell test_voxels/voxelize_mesh -->

### Boundary edges, a loss

Listing the edges on a mesh's boundary is a single pass that serial C++ finishes in less time than
a GPU round trip takes. ordito loses this one, as it does most operations whose whole cost is a few
tens of microseconds; the module pages show the rest.

<!-- bench:cell test_boundary/boundary_edges -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="boundary_edges on rim_long (131 072 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.01x</text><line class="bench-grid" x1="258.4" y1="0" x2="258.4" y2="124"/><text class="bench-tick" x="258.4" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="320.8" y1="0" x2="320.8" y2="124"/><text class="bench-tick" x="320.8" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="124"/><text class="bench-tick" x="383.2" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="124"/><text class="bench-tick" x="445.6" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">1000x</text><g><title>MeshLib: 55.0 µs, 0.36x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="9.0" width="96.7" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="7.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end">MeshLib</text><text class="bench-value" x="297.7" y="18.0">0.36x · 55.0 µs</text></g><g><title>ordito (CUDA): 154.9 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="29.0" width="124.8" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="25.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="325.8" y="38.0">1x · 154.9 µs</text></g><g><title>trimesh: 17.86 ms, 115x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="253.4" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="454.4" y="58.0">115x · 17.86 ms</text></g><g><title>MeshLab: 43.46 ms, 281x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="277.6" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="65.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">MeshLab</text><text class="bench-value" x="478.6" y="78.0">281x · 43.46 ms</text></g><g><title>libigl: 43.80 ms, 283x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="277.8" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="478.8" y="98.0">283x · 43.80 ms</text></g><g><title>PyVista: 56.55 ms, 365x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="284.7" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="111.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">PyVista</text><text class="bench-value" x="485.7" y="118.0">365x · 56.55 ms</text></g></svg>
<figcaption>rim_long (131 072 faces) · time relative to ordito, log scale · [all boundary benchmarks](boundary.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | MeshLib | 55.0 µs | 0.36x |
    | ordito (CUDA) | 154.9 µs | 1x |
    | trimesh | 17.86 ms | 115x |
    | MeshLab | 43.46 ms | 281x |
    | libigl | 43.80 ms | 283x |
    | PyVista | 56.55 ms | 365x |
<!-- /bench:cell test_boundary/boundary_edges -->

## Reproduce

The suite lives in [`benchmarks/`](https://github.com/lucagrementieri/ordito/tree/main/benchmarks);
its README describes the inputs and every comparison in detail. It needs Linux, an NVIDIA GPU and
the CUDA toolkit:

```bash
git clone https://github.com/lucagrementieri/ordito
cd ordito
uv sync --group bench --no-install-package pytorch3d
uv sync --group bench

mkdir -p results
for f in benchmarks/test_*.py; do
  uv run pytest "$f" --device=cuda --bench-all-libs \
    --benchmark-json="results/$(basename "$f" .py).json" -q
done

uv run python -m benchmarks.tools.render_docs results
```

The last command redraws every chart on this page and on the module pages from your own run. Absolute times depend on the hardware; the ratios are what carries over. See
[Performance](../performance.md) for how to read a speed claim.
