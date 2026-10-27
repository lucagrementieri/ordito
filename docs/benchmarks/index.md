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
<svg class="bench-chart" width="640" height="262" viewBox="0 0 640 262" role="img" aria-label="Time of each library relative to ordito"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="244"/><text class="bench-tick" x="196.0" y="257" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="244"/><text class="bench-tick" x="274.0" y="257" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="244"/><text class="bench-tick" x="352.0" y="257" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="244"/><text class="bench-tick" x="430.0" y="257" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="244"/><text class="bench-tick" x="508.0" y="257" text-anchor="middle">1000x</text><g><title>ordito (CUDA): the reference, 1x</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x</text></g><g><title>MeshLib: 11x ordito&#x27;s time, geometric mean over 118 functions; slower than ordito in 103</title><rect class="bench-reference" x="196" y="29.0" width="158.2" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="359.2" y="38.0">11x  (118)</text></g><g><title>PyTorch3D (CUDA): 12x ordito&#x27;s time, geometric mean over 36 functions; slower than ordito in 34</title><rect class="bench-reference" x="196" y="49.0" width="161.0" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="45.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="362.0" y="58.0">12x  (36)</text></g><g><title>NumPy: 12x ordito&#x27;s time, geometric mean over 35 functions; slower than ordito in 29</title><rect class="bench-reference" x="196" y="69.0" width="162.3" height="10"/><image class="bench-logo bench-logo-numpy" href="../assets/logos/numpy.png" x="172.6" y="65.0" width="16.9" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">NumPy</text><text class="bench-value" x="363.3" y="78.0">12x  (35)</text></g><g><title>Open3D: 22x ordito&#x27;s time, geometric mean over 71 functions; slower than ordito in 67</title><rect class="bench-reference" x="196" y="89.0" width="182.9" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="86.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="98.0" text-anchor="end">Open3D</text><text class="bench-value" x="383.9" y="98.0">22x  (71)</text></g><g><title>SciPy: 44x ordito&#x27;s time, geometric mean over 23 functions; slower than ordito in 23</title><rect class="bench-reference" x="196" y="109.0" width="206.4" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="105.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="118.0" text-anchor="end">SciPy</text><text class="bench-value" x="407.4" y="118.0">44x  (23)</text></g><g><title>potpourri3d: 65x ordito&#x27;s time, geometric mean over 20 functions; slower than ordito in 20</title><rect class="bench-reference" x="196" y="129.0" width="219.6" height="10"/><text class="bench-label" x="166" y="138.0" text-anchor="end">potpourri3d</text><text class="bench-value" x="420.6" y="138.0">65x  (20)</text></g><g><title>libigl: 95x ordito&#x27;s time, geometric mean over 95 functions; slower than ordito in 94</title><rect class="bench-reference" x="196" y="149.0" width="232.4" height="10"/><text class="bench-label" x="166" y="158.0" text-anchor="end">libigl</text><text class="bench-value" x="433.4" y="158.0">95x  (95)</text></g><g><title>MeshLab: 96x ordito&#x27;s time, geometric mean over 79 functions; slower than ordito in 79</title><rect class="bench-reference" x="196" y="169.0" width="232.7" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="165.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="178.0" text-anchor="end">MeshLab</text><text class="bench-value" x="433.7" y="178.0">96x  (79)</text></g><g><title>trimesh: 102x ordito&#x27;s time, geometric mean over 119 functions; slower than ordito in 116</title><rect class="bench-reference" x="196" y="189.0" width="234.7" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="190.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="198.0" text-anchor="end">trimesh</text><text class="bench-value" x="435.7" y="198.0">102x  (119)</text></g><g><title>PyVista: 109x ordito&#x27;s time, geometric mean over 64 functions; slower than ordito in 62</title><rect class="bench-reference" x="196" y="209.0" width="237.0" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="211.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="218.0" text-anchor="end">PyVista</text><text class="bench-value" x="438.0" y="218.0">109x  (64)</text></g><g><title>PyMeshFix: 548x ordito&#x27;s time, geometric mean over 4 functions; slower than ordito in 4</title><rect class="bench-reference" x="196" y="229.0" width="291.6" height="10"/><text class="bench-label" x="166" y="238.0" text-anchor="end">PyMeshFix</text><text class="bench-value" x="492.6" y="238.0">548x  (4)</text></g></svg>
<figcaption>Time relative to ordito (CUDA), geometric mean over every function both implement; in parentheses, the number of functions. Log scale. Runs stopped at the time limit are left out.</figcaption>
</figure>

??? note "Values"

    | Library | Time relative to ordito | Functions compared | ordito faster in |
    |---|---|---|---|
    | MeshLib | 11x | 118 | 103 |
    | PyTorch3D (CUDA) | 12x | 36 | 34 |
    | NumPy | 12x | 35 | 29 |
    | Open3D | 22x | 71 | 67 |
    | SciPy | 44x | 23 | 23 |
    | potpourri3d | 65x | 20 | 20 |
    | libigl | 95x | 95 | 94 |
    | MeshLab | 96x | 79 | 79 |
    | trimesh | 102x | 119 | 116 |
    | PyVista | 109x | 64 | 62 |
    | PyMeshFix | 548x | 4 | 4 |
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
<svg class="bench-chart" width="640" height="922" viewBox="0 0 640 922" role="img" aria-label="ordito&#x27;s speedup per module"><line class="bench-baseline" x1="196.0" y1="0" x2="196.0" y2="904"/><text class="bench-tick" x="196.0" y="917" text-anchor="middle">1x</text><line class="bench-grid" x1="300.0" y1="0" x2="300.0" y2="904"/><text class="bench-tick" x="300.0" y="917" text-anchor="middle">10x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="904"/><text class="bench-tick" x="404.0" y="917" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="904"/><text class="bench-tick" x="508.0" y="917" text-anchor="middle">1000x</text><g><title>adjacency: ordito 187x faster than the fastest library, geometric mean over 7 functions; faster in 7</title><rect class="bench-subject" x="196" y="9.0" width="236.4" height="10"/><text class="bench-label" x="190" y="18.0" text-anchor="end">adjacency</text><text class="bench-value" x="437.4" y="18.0">187x  (7)</text></g><g><title>triangles: ordito 160x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="29.0" width="229.1" height="10"/><text class="bench-label" x="190" y="38.0" text-anchor="end">triangles</text><text class="bench-value" x="430.1" y="38.0">160x  (9)</text></g><g><title>tangent_space: ordito 138x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="49.0" width="222.4" height="10"/><text class="bench-label" x="190" y="58.0" text-anchor="end">tangent_space</text><text class="bench-value" x="423.4" y="58.0">138x  (2)</text></g><g><title>geodesic_walk: ordito 110x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="69.0" width="212.3" height="10"/><text class="bench-label" x="190" y="78.0" text-anchor="end">geodesic_walk</text><text class="bench-value" x="413.3" y="78.0">110x  (5)</text></g><g><title>graph: ordito 107x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="89.0" width="211.3" height="10"/><text class="bench-label" x="190" y="98.0" text-anchor="end">graph</text><text class="bench-value" x="412.3" y="98.0">107x  (6)</text></g><g><title>grouping: ordito 88x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="109.0" width="202.2" height="10"/><text class="bench-label" x="190" y="118.0" text-anchor="end">grouping</text><text class="bench-value" x="403.2" y="118.0">88x  (4)</text></g><g><title>edges: ordito 85x faster than the fastest library, geometric mean over 14 functions; faster in 14</title><rect class="bench-subject" x="196" y="129.0" width="200.9" height="10"/><text class="bench-label" x="190" y="138.0" text-anchor="end">edges</text><text class="bench-value" x="401.9" y="138.0">85x  (14)</text></g><g><title>validation: ordito 84x faster than the fastest library, geometric mean over 11 functions; faster in 11</title><rect class="bench-subject" x="196" y="149.0" width="199.9" height="10"/><text class="bench-label" x="190" y="158.0" text-anchor="end">validation</text><text class="bench-value" x="400.9" y="158.0">84x  (11)</text></g><g><title>smoothing: ordito 72x faster than the fastest library, geometric mean over 16 functions; faster in 16</title><rect class="bench-subject" x="196" y="169.0" width="193.4" height="10"/><text class="bench-label" x="190" y="178.0" text-anchor="end">smoothing</text><text class="bench-value" x="394.4" y="178.0">72x  (16)</text></g><g><title>seams: ordito 56x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="189.0" width="181.5" height="10"/><text class="bench-label" x="190" y="198.0" text-anchor="end">seams</text><text class="bench-value" x="382.5" y="198.0">56x  (4)</text></g><g><title>curvature: ordito 54x faster than the fastest library, geometric mean over 3 functions; faster in 3</title><rect class="bench-subject" x="196" y="209.0" width="180.1" height="10"/><text class="bench-label" x="190" y="218.0" text-anchor="end">curvature</text><text class="bench-value" x="381.1" y="218.0">54x  (3)</text></g><g><title>repair: ordito 45x faster than the fastest library, geometric mean over 19 functions; faster in 17</title><rect class="bench-subject" x="196" y="229.0" width="171.5" height="10"/><text class="bench-label" x="190" y="238.0" text-anchor="end">repair</text><text class="bench-value" x="372.5" y="238.0">45x  (19)</text></g><g><title>laplacian: ordito 40x faster than the fastest library, geometric mean over 9 functions; faster in 9</title><rect class="bench-subject" x="196" y="249.0" width="166.9" height="10"/><text class="bench-label" x="190" y="258.0" text-anchor="end">laplacian</text><text class="bench-value" x="367.9" y="258.0">40x  (9)</text></g><g><title>visibility: ordito 37x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="269.0" width="163.2" height="10"/><text class="bench-label" x="190" y="278.0" text-anchor="end">visibility</text><text class="bench-value" x="364.2" y="278.0">37x  (6)</text></g><g><title>metrics: ordito 36x faster than the fastest library, geometric mean over 7 functions; faster in 7</title><rect class="bench-subject" x="196" y="289.0" width="161.5" height="10"/><text class="bench-label" x="190" y="298.0" text-anchor="end">metrics</text><text class="bench-value" x="362.5" y="298.0">36x  (7)</text></g><g><title>levelset: ordito 33x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="309.0" width="158.1" height="10"/><text class="bench-label" x="190" y="318.0" text-anchor="end">levelset</text><text class="bench-value" x="359.1" y="318.0">33x  (4)</text></g><g><title>measures: ordito 30x faster than the fastest library, geometric mean over 3 functions; faster in 3</title><rect class="bench-subject" x="196" y="329.0" width="154.1" height="10"/><text class="bench-label" x="190" y="338.0" text-anchor="end">measures</text><text class="bench-value" x="355.1" y="338.0">30x  (3)</text></g><g><title>mesh: ordito 27x faster than the fastest library, geometric mean over 13 functions; faster in 13</title><rect class="bench-subject" x="196" y="349.0" width="149.7" height="10"/><text class="bench-label" x="190" y="358.0" text-anchor="end">mesh</text><text class="bench-value" x="350.7" y="358.0">27x  (13)</text></g><g><title>energies: ordito 27x faster than the fastest library, geometric mean over 8 functions; faster in 8</title><rect class="bench-subject" x="196" y="369.0" width="148.7" height="10"/><text class="bench-label" x="190" y="378.0" text-anchor="end">energies</text><text class="bench-value" x="349.7" y="378.0">27x  (8)</text></g><g><title>interpolation: ordito 27x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="389.0" width="148.0" height="10"/><text class="bench-label" x="190" y="398.0" text-anchor="end">interpolation</text><text class="bench-value" x="349.0" y="398.0">27x  (5)</text></g><g><title>sample: ordito 24x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="409.0" width="144.4" height="10"/><text class="bench-label" x="190" y="418.0" text-anchor="end">sample</text><text class="bench-value" x="345.4" y="418.0">24x  (2)</text></g><g><title>vertices: ordito 24x faster than the fastest library, geometric mean over 4 functions; faster in 4</title><rect class="bench-subject" x="196" y="429.0" width="143.2" height="10"/><text class="bench-label" x="190" y="438.0" text-anchor="end">vertices</text><text class="bench-value" x="344.2" y="438.0">24x  (4)</text></g><g><title>remesh: ordito 24x faster than the fastest library, geometric mean over 10 functions; faster in 10</title><rect class="bench-subject" x="196" y="449.0" width="142.7" height="10"/><text class="bench-label" x="190" y="458.0" text-anchor="end">remesh</text><text class="bench-value" x="343.7" y="458.0">24x  (10)</text></g><g><title>holes: ordito 21x faster than the fastest library, geometric mean over 12 functions; faster in 12</title><rect class="bench-subject" x="196" y="469.0" width="138.2" height="10"/><text class="bench-label" x="190" y="478.0" text-anchor="end">holes</text><text class="bench-value" x="339.2" y="478.0">21x  (12)</text></g><g><title>neighbors: ordito 21x faster than the fastest library, geometric mean over 14 functions; faster in 14</title><rect class="bench-subject" x="196" y="489.0" width="136.9" height="10"/><text class="bench-label" x="190" y="498.0" text-anchor="end">neighbors</text><text class="bench-value" x="337.9" y="498.0">21x  (14)</text></g><g><title>points: ordito 19x faster than the fastest library, geometric mean over 19 functions; faster in 19</title><rect class="bench-subject" x="196" y="509.0" width="134.2" height="10"/><text class="bench-label" x="190" y="518.0" text-anchor="end">points</text><text class="bench-value" x="335.2" y="518.0">19x  (19)</text></g><g><title>creation: ordito 18x faster than the fastest library, geometric mean over 20 functions; faster in 16</title><rect class="bench-subject" x="196" y="529.0" width="131.0" height="10"/><text class="bench-label" x="190" y="538.0" text-anchor="end">creation</text><text class="bench-value" x="332.0" y="538.0">18x  (20)</text></g><g><title>intersection: ordito 18x faster than the fastest library, geometric mean over 11 functions; faster in 9</title><rect class="bench-subject" x="196" y="549.0" width="129.8" height="10"/><text class="bench-label" x="190" y="558.0" text-anchor="end">intersection</text><text class="bench-value" x="330.8" y="558.0">18x  (11)</text></g><g><title>heat: ordito 17x faster than the fastest library, geometric mean over 8 functions; faster in 8</title><rect class="bench-subject" x="196" y="569.0" width="127.9" height="10"/><text class="bench-label" x="190" y="578.0" text-anchor="end">heat</text><text class="bench-value" x="328.9" y="578.0">17x  (8)</text></g><g><title>array: ordito 17x faster than the fastest library, geometric mean over 20 functions; faster in 18</title><rect class="bench-subject" x="196" y="589.0" width="127.2" height="10"/><text class="bench-label" x="190" y="598.0" text-anchor="end">array</text><text class="bench-value" x="328.2" y="598.0">17x  (20)</text></g><g><title>transform: ordito 16x faster than the fastest library, geometric mean over 5 functions; faster in 3</title><rect class="bench-subject" x="196" y="609.0" width="124.7" height="10"/><text class="bench-label" x="190" y="618.0" text-anchor="end">transform</text><text class="bench-value" x="325.7" y="618.0">16x  (5)</text></g><g><title>parametrization: ordito 10x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="629.0" width="103.9" height="10"/><text class="bench-label" x="190" y="638.0" text-anchor="end">parametrization</text><text class="bench-value" x="304.9" y="638.0">10x  (6)</text></g><g><title>selection: ordito 9x faster than the fastest library, geometric mean over 8 functions; faster in 5</title><rect class="bench-subject" x="196" y="649.0" width="99.4" height="10"/><text class="bench-label" x="190" y="658.0" text-anchor="end">selection</text><text class="bench-value" x="300.4" y="658.0">9x  (8)</text></g><g><title>boundary: ordito 8.8x faster than the fastest library, geometric mean over 7 functions; faster in 5</title><rect class="bench-subject" x="196" y="669.0" width="98.0" height="10"/><text class="bench-label" x="190" y="678.0" text-anchor="end">boundary</text><text class="bench-value" x="299.0" y="678.0">8.8x  (7)</text></g><g><title>ray: ordito 8.7x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="689.0" width="97.7" height="10"/><text class="bench-label" x="190" y="698.0" text-anchor="end">ray</text><text class="bench-value" x="298.7" y="698.0">8.7x  (5)</text></g><g><title>registration: ordito 8.6x faster than the fastest library, geometric mean over 5 functions; faster in 5</title><rect class="bench-subject" x="196" y="709.0" width="97.0" height="10"/><text class="bench-label" x="190" y="718.0" text-anchor="end">registration</text><text class="bench-value" x="298.0" y="718.0">8.6x  (5)</text></g><g><title>proximity: ordito 7.2x faster than the fastest library, geometric mean over 6 functions; faster in 5</title><rect class="bench-subject" x="196" y="729.0" width="88.9" height="10"/><text class="bench-label" x="190" y="738.0" text-anchor="end">proximity</text><text class="bench-value" x="289.9" y="738.0">7.2x  (6)</text></g><g><title>combine: ordito 6.9x faster than the fastest library, geometric mean over 2 functions; faster in 2</title><rect class="bench-subject" x="196" y="749.0" width="87.2" height="10"/><text class="bench-label" x="190" y="758.0" text-anchor="end">combine</text><text class="bench-value" x="288.2" y="758.0">6.9x  (2)</text></g><g><title>voxels: ordito 6.3x faster than the fastest library, geometric mean over 15 functions; faster in 12</title><rect class="bench-subject" x="196" y="769.0" width="83.1" height="10"/><text class="bench-label" x="190" y="778.0" text-anchor="end">voxels</text><text class="bench-value" x="284.1" y="778.0">6.3x  (15)</text></g><g><title>polyline: ordito 4.6x faster than the fastest library, geometric mean over 7 functions; faster in 5</title><rect class="bench-subject" x="196" y="789.0" width="68.5" height="10"/><text class="bench-label" x="190" y="798.0" text-anchor="end">polyline</text><text class="bench-value" x="269.5" y="798.0">4.6x  (7)</text></g><g><title>reconstruction: ordito 3.8x faster than the fastest library, geometric mean over 6 functions; faster in 6</title><rect class="bench-subject" x="196" y="809.0" width="60.6" height="10"/><text class="bench-label" x="190" y="818.0" text-anchor="end">reconstruction</text><text class="bench-value" x="261.6" y="818.0">3.8x  (6)</text></g><g><title>homology: ordito 3.2x faster than the fastest library, geometric mean over 1 functions; faster in 1</title><rect class="bench-subject" x="196" y="829.0" width="52.3" height="10"/><text class="bench-label" x="190" y="838.0" text-anchor="end">homology</text><text class="bench-value" x="253.3" y="838.0">3.2x  (1)</text></g><g><title>linalg: ordito 3.2x faster than the fastest library, geometric mean over 1 functions; faster in 1</title><rect class="bench-subject" x="196" y="849.0" width="52.0" height="10"/><text class="bench-label" x="190" y="858.0" text-anchor="end">linalg</text><text class="bench-value" x="253.0" y="858.0">3.2x  (1)</text></g><g><title>reduce: ordito 3.1x faster than the fastest library, geometric mean over 15 functions; faster in 9</title><rect class="bench-subject" x="196" y="869.0" width="51.3" height="10"/><text class="bench-label" x="190" y="878.0" text-anchor="end">reduce</text><text class="bench-value" x="252.3" y="878.0">3.1x  (15)</text></g><g><title>bounds: ordito 1.3x faster than the fastest library, geometric mean over 9 functions; faster in 7</title><rect class="bench-subject" x="196" y="889.0" width="11.6" height="10"/><text class="bench-label" x="190" y="898.0" text-anchor="end">bounds</text><text class="bench-value" x="212.6" y="898.0">1.3x  (9)</text></g></svg>
<figcaption>ordito (CUDA)'s speedup over the fastest library measured on each input, geometric mean per module; in parentheses, the number of functions. Log scale.</figcaption>
</figure>

??? note "Values"

    | Module | ordito's speedup | Functions | ordito faster in |
    |---|---|---|---|
    | [`adjacency`](adjacency.md) | 187x | 7 | 7 |
    | [`triangles`](triangles.md) | 160x | 9 | 9 |
    | [`tangent_space`](tangent_space.md) | 138x | 2 | 2 |
    | [`geodesic_walk`](geodesic_walk.md) | 110x | 5 | 5 |
    | [`graph`](graph.md) | 107x | 6 | 6 |
    | [`grouping`](grouping.md) | 88x | 4 | 4 |
    | [`edges`](edges.md) | 85x | 14 | 14 |
    | [`validation`](validation.md) | 84x | 11 | 11 |
    | [`smoothing`](smoothing.md) | 72x | 16 | 16 |
    | [`seams`](seams.md) | 56x | 4 | 4 |
    | [`curvature`](curvature.md) | 54x | 3 | 3 |
    | [`repair`](repair.md) | 45x | 19 | 17 |
    | [`laplacian`](laplacian.md) | 40x | 9 | 9 |
    | [`visibility`](visibility.md) | 37x | 6 | 6 |
    | [`metrics`](metrics.md) | 36x | 7 | 7 |
    | [`levelset`](levelset.md) | 33x | 4 | 4 |
    | [`measures`](measures.md) | 30x | 3 | 3 |
    | [`mesh`](mesh.md) | 27x | 13 | 13 |
    | [`energies`](energies.md) | 27x | 8 | 8 |
    | [`interpolation`](interpolation.md) | 27x | 5 | 5 |
    | [`sample`](sample.md) | 24x | 2 | 2 |
    | [`vertices`](vertices.md) | 24x | 4 | 4 |
    | [`remesh`](remesh.md) | 24x | 10 | 10 |
    | [`holes`](holes.md) | 21x | 12 | 12 |
    | [`neighbors`](neighbors.md) | 21x | 14 | 14 |
    | [`points`](points.md) | 19x | 19 | 19 |
    | [`creation`](creation.md) | 18x | 20 | 16 |
    | [`intersection`](intersection.md) | 18x | 11 | 9 |
    | [`heat`](heat.md) | 17x | 8 | 8 |
    | [`array`](array.md) | 17x | 20 | 18 |
    | [`transform`](transform.md) | 16x | 5 | 3 |
    | [`parametrization`](parametrization.md) | 10x | 6 | 6 |
    | [`selection`](selection.md) | 9x | 8 | 5 |
    | [`boundary`](boundary.md) | 8.8x | 7 | 5 |
    | [`ray`](ray.md) | 8.7x | 5 | 5 |
    | [`registration`](registration.md) | 8.6x | 5 | 5 |
    | [`proximity`](proximity.md) | 7.2x | 6 | 5 |
    | [`combine`](combine.md) | 6.9x | 2 | 2 |
    | [`voxels`](voxels.md) | 6.3x | 15 | 12 |
    | [`polyline`](polyline.md) | 4.6x | 7 | 5 |
    | [`reconstruction`](reconstruction.md) | 3.8x | 6 | 6 |
    | [`homology`](homology.md) | 3.2x | 1 | 1 |
    | [`linalg`](linalg.md) | 3.2x | 1 | 1 |
    | [`reduce`](reduce.md) | 3.1x | 15 | 9 |
    | [`bounds`](bounds.md) | 1.3x | 9 | 7 |
<!-- /bench:modules -->

## Highlights

The comparisons with the widest field of libraries. Each chart shows one input, with ordito at 1x
and every other library's time relative to it; each module's page has every function and every
input.

### Signed distance to a mesh

Signed distance from a point set to a triangle mesh, the widest field among the spatial queries.

<!-- bench:cell test_proximity/signed_distance_on_mesh -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="signed_distance_on_mesh on dragon (871 414 faces) · sign_mode=parity"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="124"/><text class="bench-tick" x="258.4" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="124"/><text class="bench-tick" x="320.8" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="124"/><text class="bench-tick" x="383.2" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="124"/><text class="bench-tick" x="445.6" y="137" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 5.11 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 5.11 ms</text></g><g><title>MeshLib: 94.85 ms, 19x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="141.6" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="342.6" y="38.0">19x · 94.85 ms</text></g><g><title>Open3D: 281.78 ms, 55x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="171.1" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="372.1" y="58.0">55x · 281.78 ms</text></g><g><title>PyVista: 1.024 s, 200x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="206.0" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="407.0" y="78.0">200x · 1.024 s</text></g><g><title>libigl: 1.045 s, 205x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="206.6" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="407.6" y="98.0">205x · 1.045 s</text></g><g><title>MeshLab: 7.216 s, 1413x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="259.0" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="105.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">MeshLab</text><text class="bench-value" x="460.0" y="118.0">1413x · 7.216 s</text></g></svg>
<figcaption>dragon (871 414 faces) · sign_mode=parity · time relative to ordito, log scale · [all proximity benchmarks](proximity.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 5.11 ms | 1x |
    | MeshLib | 94.85 ms | 19x |
    | Open3D | 281.78 ms | 55x |
    | PyVista | 1.024 s | 200x |
    | libigl | 1.045 s | 205x |
    | MeshLab | 7.216 s | 1413x |
<!-- /bench:cell test_proximity/signed_distance_on_mesh -->

### Nearest neighbours, k = 7

Seven nearest neighbours of every query point, against CPU trees and PyTorch3D's GPU kernel.

<!-- bench:cell test_neighbors/query_nearest_bvh_k7 -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="query_nearest_bvh_k7 on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="104"/><text class="bench-tick" x="300.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="104"/><text class="bench-tick" x="404.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">100x</text><g><title>ordito (CUDA): 1.85 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 1.85 ms</text></g><g><title>PyTorch3D (CUDA): 43.89 ms, 24x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="247.0" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="25.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="448.0" y="38.0">24x · 43.89 ms</text></g><g><title>Open3D: 117.91 ms, 64x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="291.7" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="492.7" y="58.0">64x · 117.91 ms</text></g><g><title>SciPy: 147.45 ms, 80x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="301.8" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="65.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="78.0" text-anchor="end">SciPy</text><text class="bench-value" x="502.8" y="78.0">80x · 147.45 ms</text></g><g><title>libigl: out of memory</title><path class="bench-failed" d="M198,90.0L206,98.0M206,90.0L198,98.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="212.0" y="98.0">out of memory</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · a red cross is a run that left no time · [all neighbors benchmarks](neighbors.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.85 ms | 1x |
    | PyTorch3D (CUDA) | 43.89 ms | 24x |
    | Open3D | 117.91 ms | 64x |
    | SciPy | 147.45 ms | 80x |
    | libigl | out of memory | |
<!-- /bench:cell test_neighbors/query_nearest_bvh_k7 -->

### Nearest neighbour, k = 1

The same query with a single neighbour. Every library with a tree, ordito included, builds it
over the cloud inside the timed call, and the build is most of the work: about two thirds of
ordito's time.

<!-- bench:cell test_neighbors/query_nearest_bvh_k1 -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="query_nearest_bvh_k1 on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="124"/><text class="bench-tick" x="274.0" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="124"/><text class="bench-tick" x="352.0" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="124"/><text class="bench-tick" x="430.0" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 625.7 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 625.7 µs</text></g><g><title>MeshLib: 15.65 ms, 25x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="187.0" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="388.0" y="38.0">25x · 15.65 ms</text></g><g><title>PyTorch3D (CUDA): 33.85 ms, 54x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="213.2" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="45.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="414.2" y="58.0">54x · 33.85 ms</text></g><g><title>Open3D: 125.54 ms, 201x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="257.6" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="458.6" y="78.0">201x · 125.54 ms</text></g><g><title>SciPy: 167.32 ms, 267x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="267.3" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="85.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="98.0" text-anchor="end">SciPy</text><text class="bench-value" x="468.3" y="98.0">267x · 167.32 ms</text></g><g><title>libigl: out of memory</title><path class="bench-failed" d="M198,110.0L206,118.0M206,110.0L198,118.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">libigl</text><text class="bench-value" x="212.0" y="118.0">out of memory</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · a red cross is a run that left no time · [all neighbors benchmarks](neighbors.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 625.7 µs | 1x |
    | MeshLib | 15.65 ms | 25x |
    | PyTorch3D (CUDA) | 33.85 ms | 54x |
    | Open3D | 125.54 ms | 201x |
    | SciPy | 167.32 ms | 267x |
    | libigl | out of memory | |
<!-- /bench:cell test_neighbors/query_nearest_bvh_k1 -->

### Quadric decimation

Edge-collapse simplification under the quadric error metric, against five independent
decimators.

<!-- bench:cell test_remesh/quadric_decimate -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="quadric_decimate on saddle (34 848 faces) · target_ratio=0.5"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="124"/><text class="bench-tick" x="300.0" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="124"/><text class="bench-tick" x="404.0" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">100x</text><g><title>ordito (CUDA): 2.99 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 2.99 ms</text></g><g><title>MeshLib: 19.08 ms, 6.4x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="187.7" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="388.7" y="38.0">6.4x · 19.08 ms</text></g><g><title>Open3D: 45.93 ms, 15x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="227.4" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="428.4" y="58.0">15x · 45.93 ms</text></g><g><title>PyVista: 46.34 ms, 15x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="227.8" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="428.8" y="78.0">15x · 46.34 ms</text></g><g><title>libigl: 51.05 ms, 17x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="232.1" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="433.1" y="98.0">17x · 51.05 ms</text></g><g><title>MeshLab: 177.33 ms, 59x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="288.4" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="105.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">MeshLab</text><text class="bench-value" x="489.4" y="118.0">59x · 177.33 ms</text></g></svg>
<figcaption>saddle (34 848 faces) · target_ratio=0.5 · time relative to ordito, log scale · [all remesh benchmarks](remesh.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 2.99 ms | 1x |
    | MeshLib | 19.08 ms | 6.4x |
    | Open3D | 45.93 ms | 15x |
    | PyVista | 46.34 ms | 15x |
    | libigl | 51.05 ms | 17x |
    | MeshLab | 177.33 ms | 59x |
<!-- /bench:cell test_remesh/quadric_decimate -->

### Minimal oriented bounding box

Every library here searches for the minimum-volume box, not the PCA box, so the bars compare
the same optimisation.

<!-- bench:cell test_bounds/oriented_bounding_box -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="oriented_bounding_box on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 1.58 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 1.58 ms</text></g><g><title>PyVista: 18.34 ms, 12x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="128.8" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="329.8" y="38.0">12x · 18.34 ms</text></g><g><title>trimesh: 159.09 ms, 100x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="187.3" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="388.3" y="58.0">100x · 159.09 ms</text></g><g><title>Open3D: 2.096 s, 1322x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="257.2" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="458.2" y="78.0">1322x · 2.096 s</text></g><g><title>libigl: 2.483 s, 1567x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="261.8" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="462.8" y="98.0">1567x · 2.483 s</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all bounds benchmarks](bounds.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.58 ms | 1x |
    | PyVista | 18.34 ms | 12x |
    | trimesh | 159.09 ms | 100x |
    | Open3D | 2.096 s | 1322x |
    | libigl | 2.483 s | 1567x |
<!-- /bench:cell test_bounds/oriented_bounding_box -->

### Convex hull

<!-- bench:cell test_points/convex_hull -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="convex_hull on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 1.86 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 1.86 ms</text></g><g><title>Open3D: 49.37 ms, 27x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="189.1" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="26.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="38.0" text-anchor="end">Open3D</text><text class="bench-value" x="390.1" y="38.0">27x · 49.37 ms</text></g><g><title>SciPy: 61.31 ms, 33x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="196.4" height="10"/><image class="bench-logo bench-logo-scipy" href="../assets/logos/scipy.png" x="172.0" y="45.4" width="18.0" height="17.1"/><text class="bench-label" x="166" y="58.0" text-anchor="end">SciPy</text><text class="bench-value" x="397.4" y="58.0">33x · 61.31 ms</text></g><g><title>trimesh: 79.55 ms, 43x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="205.2" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="70.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="78.0" text-anchor="end">trimesh</text><text class="bench-value" x="406.2" y="78.0">43x · 79.55 ms</text></g><g><title>MeshLab: 417.44 ms, 224x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="261.4" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="462.4" y="98.0">224x · 417.44 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all points benchmarks](points.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.86 ms | 1x |
    | Open3D | 49.37 ms | 27x |
    | SciPy | 61.31 ms | 33x |
    | trimesh | 79.55 ms | 43x |
    | MeshLab | 417.44 ms | 224x |
<!-- /bench:cell test_points/convex_hull -->

### Blue-noise sampling

Poisson-disk sampling of a surface: four different algorithms for the same distribution.

<!-- bench:cell test_sample/blue_noise -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="blue_noise on dragon (871 414 faces) · radius_scale=1.0"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 1.21 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 1.21 ms</text></g><g><title>MeshLib: 12.08 ms, 10x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="156.0" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="357.0" y="38.0">10x · 12.08 ms</text></g><g><title>Open3D: 57.02 ms, 47x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="208.5" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="409.5" y="58.0">47x · 57.02 ms</text></g><g><title>libigl: 304.39 ms, 252x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="265.3" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="466.3" y="78.0">252x · 304.39 ms</text></g><g><title>MeshLab: 488.56 ms, 404x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="281.3" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="482.3" y="98.0">404x · 488.56 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · radius_scale=1.0 · time relative to ordito, log scale · [all sample benchmarks](sample.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 1.21 ms | 1x |
    | MeshLib | 12.08 ms | 10x |
    | Open3D | 57.02 ms | 47x |
    | libigl | 304.39 ms | 252x |
    | MeshLab | 488.56 ms | 404x |
<!-- /bench:cell test_sample/blue_noise -->

### Geodesic distance

Geodesic distance from a source vertex by the heat method, including building the solver.

<!-- bench:cell test_heat/heat_geodesic -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="heat_geodesic on sphere_med (81 920 faces) · setup=full"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="300.0" y1="0" x2="300.0" y2="104"/><text class="bench-tick" x="300.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="404.0" y1="0" x2="404.0" y2="104"/><text class="bench-tick" x="404.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">100x</text><g><title>ordito (CUDA): 6.68 ms, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="104.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="305.0" y="18.0">1x · 6.68 ms</text></g><g><title>PyVista: 23.72 ms, 3.5x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="161.2" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="362.2" y="38.0">3.5x · 23.72 ms</text></g><g><title>potpourri3d: 539.58 ms, 81x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="302.3" height="10"/><text class="bench-label" x="166" y="58.0" text-anchor="end">potpourri3d</text><text class="bench-value" x="503.3" y="58.0">81x · 539.58 ms</text></g><g><title>libigl: 597.39 ms, 89x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="306.9" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="507.9" y="78.0">89x · 597.39 ms</text></g><g><title>MeshLab: 601.38 ms, 90x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="307.2" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="508.2" y="98.0">90x · 601.38 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · setup=full · time relative to ordito, log scale · [all heat benchmarks](heat.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 6.68 ms | 1x |
    | PyVista | 23.72 ms | 3.5x |
    | potpourri3d | 539.58 ms | 81x |
    | libigl | 597.39 ms | 89x |
    | MeshLab | 601.38 ms | 90x |
<!-- /bench:cell test_heat/heat_geodesic -->

### Repair: merging duplicate vertices

<!-- bench:cell test_repair/remove_duplicated_vertices -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="remove_duplicated_vertices on sphere_med (81 920 faces) · epsilon=0.0"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 218.1 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 218.1 µs</text></g><g><title>Open3D: 16.42 ms, 75x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="179.5" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="26.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="38.0" text-anchor="end">Open3D</text><text class="bench-value" x="380.5" y="38.0">75x · 16.42 ms</text></g><g><title>PyVista: 25.66 ms, 118x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="191.6" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="51.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyVista</text><text class="bench-value" x="392.6" y="58.0">118x · 25.66 ms</text></g><g><title>MeshLib: 31.33 ms, 144x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="197.0" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="67.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">MeshLib</text><text class="bench-value" x="398.0" y="78.0">144x · 31.33 ms</text></g><g><title>MeshLab: 221.69 ms, 1017x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="250.0" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="451.0" y="98.0">1017x · 221.69 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · epsilon=0.0 · time relative to ordito, log scale · [all repair benchmarks](repair.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 218.1 µs | 1x |
    | Open3D | 16.42 ms | 75x |
    | PyVista | 25.66 ms | 118x |
    | MeshLib | 31.33 ms | 144x |
    | MeshLab | 221.69 ms | 1017x |
<!-- /bench:cell test_repair/remove_duplicated_vertices -->

### Repair: consistent winding

Re-orienting every face to agree with its neighbours, a whole-mesh propagation.

<!-- bench:cell test_repair/make_winding_consistent -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="make_winding_consistent on sphere_med (81 920 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 148.6 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 148.6 µs</text></g><g><title>MeshLib: 6.91 ms, 46x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="208.1" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="409.1" y="38.0">46x · 6.91 ms</text></g><g><title>trimesh: 17.97 ms, 121x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="240.4" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="441.4" y="58.0">121x · 17.97 ms</text></g><g><title>libigl: 36.63 ms, 247x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="264.6" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="465.6" y="78.0">247x · 36.63 ms</text></g><g><title>MeshLab: 51.21 ms, 345x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="275.9" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="476.9" y="98.0">345x · 51.21 ms</text></g></svg>
<figcaption>sphere_med (81 920 faces) · time relative to ordito, log scale · [all repair benchmarks](repair.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 148.6 µs | 1x |
    | MeshLib | 6.91 ms | 46x |
    | trimesh | 17.97 ms | 121x |
    | libigl | 36.63 ms | 247x |
    | MeshLab | 51.21 ms | 345x |
<!-- /bench:cell test_repair/make_winding_consistent -->

### Connected components

<!-- bench:cell test_adjacency/face_connected_component_labels -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="102" viewBox="0 0 640 102" role="img" aria-label="face_connected_component_labels on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="84"/><text class="bench-tick" x="196.0" y="97" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="84"/><text class="bench-tick" x="258.4" y="97" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="84"/><text class="bench-tick" x="320.8" y="97" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="84"/><text class="bench-tick" x="383.2" y="97" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="84"/><text class="bench-tick" x="445.6" y="97" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="84"/><text class="bench-tick" x="508.0" y="97" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 301.0 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 301.0 µs</text></g><g><title>MeshLib: 21.73 ms, 72x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="178.4" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="379.4" y="38.0">72x · 21.73 ms</text></g><g><title>PyVista: 318.91 ms, 1060x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="251.2" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="51.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="58.0" text-anchor="end">PyVista</text><text class="bench-value" x="452.2" y="58.0">1060x · 318.91 ms</text></g><g><title>libigl: 445.32 ms, 1480x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="260.2" height="10"/><text class="bench-label" x="166" y="78.0" text-anchor="end">libigl</text><text class="bench-value" x="461.2" y="78.0">1480x · 445.32 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all adjacency benchmarks](adjacency.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 301.0 µs | 1x |
    | MeshLib | 21.73 ms | 72x |
    | PyVista | 318.91 ms | 1060x |
    | libigl | 445.32 ms | 1480x |
<!-- /bench:cell test_adjacency/face_connected_component_labels -->

### Unique edges

The undirected edge list, the first step of most topology code.

<!-- bench:cell test_edges/edges_unique -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="edges_unique on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="274.0" y1="0" x2="274.0" y2="104"/><text class="bench-tick" x="274.0" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="352.0" y1="0" x2="352.0" y2="104"/><text class="bench-tick" x="352.0" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="430.0" y1="0" x2="430.0" y2="104"/><text class="bench-tick" x="430.0" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">1000x</text><g><title>ordito (CUDA): 472.6 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="78.0" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="279.0" y="18.0">1x · 472.6 µs</text></g><g><title>PyTorch3D (CUDA): 3.52 ms, 7.5x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="146.0" height="10"/><image class="bench-logo bench-logo-pytorch3d" href="../assets/logos/pytorch3d.png" x="173.0" y="25.0" width="15.9" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyTorch3D (CUDA)</text><text class="bench-value" x="347.0" y="38.0">7.5x · 3.52 ms</text></g><g><title>trimesh: 243.05 ms, 514x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="289.5" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="490.5" y="58.0">514x · 243.05 ms</text></g><g><title>PyVista: 268.58 ms, 568x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="292.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="493.9" y="78.0">568x · 268.58 ms</text></g><g><title>libigl: 330.25 ms, 699x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="299.9" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="500.9" y="98.0">699x · 330.25 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all edges benchmarks](edges.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 472.6 µs | 1x |
    | PyTorch3D (CUDA) | 3.52 ms | 7.5x |
    | trimesh | 243.05 ms | 514x |
    | PyVista | 268.58 ms | 568x |
    | libigl | 330.25 ms | 699x |
<!-- /bench:cell test_edges/edges_unique -->

### Taubin smoothing

<!-- bench:cell test_smoothing/filter_taubin -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="filter_taubin on dragon (871 414 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 476.5 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 476.5 µs</text></g><g><title>PyVista: 153.29 ms, 322x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="218.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="31.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">PyVista</text><text class="bench-value" x="419.9" y="38.0">322x · 153.29 ms</text></g><g><title>trimesh: 198.54 ms, 417x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="225.9" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="426.9" y="58.0">417x · 198.54 ms</text></g><g><title>Open3D: 306.02 ms, 642x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="237.6" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="66.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="78.0" text-anchor="end">Open3D</text><text class="bench-value" x="438.6" y="78.0">642x · 306.02 ms</text></g><g><title>MeshLab: 867.52 ms, 1821x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="265.8" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="85.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="98.0" text-anchor="end">MeshLab</text><text class="bench-value" x="466.8" y="98.0">1821x · 867.52 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · time relative to ordito, log scale · [all smoothing benchmarks](smoothing.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 476.5 µs | 1x |
    | PyVista | 153.29 ms | 322x |
    | trimesh | 198.54 ms | 417x |
    | Open3D | 306.02 ms | 642x |
    | MeshLab | 867.52 ms | 1821x |
<!-- /bench:cell test_smoothing/filter_taubin -->

### Voxelizing a mesh

<!-- bench:cell test_voxels/voxelize_mesh -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="122" viewBox="0 0 640 122" role="img" aria-label="voxelize_mesh on dragon (871 414 faces) · divisor=64"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="104"/><text class="bench-tick" x="196.0" y="117" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="258.4" y1="0" x2="258.4" y2="104"/><text class="bench-tick" x="258.4" y="117" text-anchor="middle">1x</text><line class="bench-grid" x1="320.8" y1="0" x2="320.8" y2="104"/><text class="bench-tick" x="320.8" y="117" text-anchor="middle">10x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="104"/><text class="bench-tick" x="383.2" y="117" text-anchor="middle">100x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="104"/><text class="bench-tick" x="445.6" y="117" text-anchor="middle">1000x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="104"/><text class="bench-tick" x="508.0" y="117" text-anchor="middle">10000x</text><g><title>ordito (CUDA): 636.0 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="9.0" width="62.4" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="5.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="263.4" y="18.0">1x · 636.0 µs</text></g><g><title>MeshLib: 109.34 ms, 172x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="29.0" width="201.9" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="27.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end">MeshLib</text><text class="bench-value" x="402.9" y="38.0">172x · 109.34 ms</text></g><g><title>Open3D: 161.89 ms, 255x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="212.5" height="10"/><image class="bench-logo bench-logo-open3d" href="../assets/logos/open3d.png" x="172.0" y="46.2" width="18.0" height="15.6"/><text class="bench-label" x="166" y="58.0" text-anchor="end">Open3D</text><text class="bench-value" x="413.5" y="58.0">255x · 161.89 ms</text></g><g><title>PyVista: 255.97 ms, 402x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="224.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="71.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">PyVista</text><text class="bench-value" x="425.9" y="78.0">402x · 255.97 ms</text></g><g><title>trimesh: 655.79 ms, 1031x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="250.4" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="90.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="98.0" text-anchor="end">trimesh</text><text class="bench-value" x="451.4" y="98.0">1031x · 655.79 ms</text></g></svg>
<figcaption>dragon (871 414 faces) · divisor=64 · time relative to ordito, log scale · [all voxels benchmarks](voxels.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | ordito (CUDA) | 636.0 µs | 1x |
    | MeshLib | 109.34 ms | 172x |
    | Open3D | 161.89 ms | 255x |
    | PyVista | 255.97 ms | 402x |
    | trimesh | 655.79 ms | 1031x |
<!-- /bench:cell test_voxels/voxelize_mesh -->

### Boundary edges, a loss

Listing the edges on a mesh's boundary is a single pass that serial C++ finishes in less time than
a GPU round trip takes. ordito loses this one, as it does most operations whose whole cost is a few
tens of microseconds; the module pages show the rest.

<!-- bench:cell test_boundary/boundary_edges -->
<figure class="bench-figure">
<svg class="bench-chart" width="640" height="142" viewBox="0 0 640 142" role="img" aria-label="boundary_edges on rim_long (131 072 faces)"><line class="bench-grid" x1="196.0" y1="0" x2="196.0" y2="124"/><text class="bench-tick" x="196.0" y="137" text-anchor="middle">0.01x</text><line class="bench-grid" x1="258.4" y1="0" x2="258.4" y2="124"/><text class="bench-tick" x="258.4" y="137" text-anchor="middle">0.1x</text><line class="bench-baseline" x1="320.8" y1="0" x2="320.8" y2="124"/><text class="bench-tick" x="320.8" y="137" text-anchor="middle">1x</text><line class="bench-grid" x1="383.2" y1="0" x2="383.2" y2="124"/><text class="bench-tick" x="383.2" y="137" text-anchor="middle">10x</text><line class="bench-grid" x1="445.6" y1="0" x2="445.6" y2="124"/><text class="bench-tick" x="445.6" y="137" text-anchor="middle">100x</text><line class="bench-grid" x1="508.0" y1="0" x2="508.0" y2="124"/><text class="bench-tick" x="508.0" y="137" text-anchor="middle">1000x</text><g><title>MeshLib: 55.0 µs, 0.31x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="9.0" width="92.9" height="10"/><image class="bench-logo bench-logo-meshlib" href="../assets/logos/meshlib.png" x="172.0" y="7.5" width="18.0" height="13.0"/><text class="bench-label" x="166" y="18.0" text-anchor="end">MeshLib</text><text class="bench-value" x="293.9" y="18.0">0.31x · 55.0 µs</text></g><g><title>ordito (CUDA): 178.5 µs, 1x ordito&#x27;s time</title><rect class="bench-subject" x="196" y="29.0" width="124.8" height="10"/><image class="bench-logo bench-logo-ordito" href="../assets/logos/ordito.png" x="172.4" y="25.0" width="17.2" height="18.0"/><text class="bench-label" x="166" y="38.0" text-anchor="end" font-weight="600">ordito (CUDA)</text><text class="bench-value" x="325.8" y="38.0">1x · 178.5 µs</text></g><g><title>trimesh: 17.86 ms, 100x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="49.0" width="249.6" height="10"/><image class="bench-logo bench-logo-trimesh" href="../assets/logos/trimesh.png" x="172.0" y="50.2" width="18.0" height="7.7"/><text class="bench-label" x="166" y="58.0" text-anchor="end">trimesh</text><text class="bench-value" x="450.6" y="58.0">100x · 17.86 ms</text></g><g><title>MeshLab: 43.46 ms, 243x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="69.0" width="273.7" height="10"/><image class="bench-logo bench-logo-meshlab" href="../assets/logos/meshlab.png" x="172.0" y="65.0" width="18.0" height="18.0"/><text class="bench-label" x="166" y="78.0" text-anchor="end">MeshLab</text><text class="bench-value" x="474.7" y="78.0">243x · 43.46 ms</text></g><g><title>libigl: 43.80 ms, 245x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="89.0" width="273.9" height="10"/><text class="bench-label" x="166" y="98.0" text-anchor="end">libigl</text><text class="bench-value" x="474.9" y="98.0">245x · 43.80 ms</text></g><g><title>PyVista: 56.55 ms, 317x ordito&#x27;s time</title><rect class="bench-reference" x="196" y="109.0" width="280.9" height="10"/><image class="bench-logo bench-logo-pyvista" href="../assets/logos/pyvista.png" x="172.0" y="111.0" width="18.0" height="6.0"/><text class="bench-label" x="166" y="118.0" text-anchor="end">PyVista</text><text class="bench-value" x="481.9" y="118.0">317x · 56.55 ms</text></g></svg>
<figcaption>rim_long (131 072 faces) · time relative to ordito, log scale · [all boundary benchmarks](boundary.md)</figcaption>
</figure>

??? note "Values"

    | Library | Median | Relative to ordito |
    |---|---|---|
    | MeshLib | 55.0 µs | 0.31x |
    | ordito (CUDA) | 178.5 µs | 1x |
    | trimesh | 17.86 ms | 100x |
    | MeshLab | 43.46 ms | 243x |
    | libigl | 43.80 ms | 245x |
    | PyVista | 56.55 ms | 317x |
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
