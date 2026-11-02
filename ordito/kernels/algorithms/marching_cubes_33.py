"""
Kernels for Lewiner's topologically controlled marching cubes (MC33) on a ``float64`` node lattice.

One cell of the lattice is classified as in Lewiner, Lopes, Vieira and Tavares, *Efficient
implementation of Marching Cubes' cases with topological guarantees* (JGT 8(2), 2003): the sign
pattern of its eight corners names one of the 15 cases of Chernyaev's MC33 and a configuration,
and the cases whose triangulation is ambiguous take a *face test* (the asymptotic decider on one
face: whether the two components touching it join through its interior) or an *interior test*
(whether they join through the cell's interior) to pick a subcase. Some subcases add one vertex
inside the cell. Every crossing edge gets a vertex interpolated linearly in the field, and the
tiling tables name the triangles over the twelve edges (``0``-``11``) and the cell vertex (``12``).

Corner ``c`` sits at offset ``(c & 1 ^ (c >> 1) & 1, (c >> 1) & 1, (c >> 2) & 1)`` of the cell's
lowest node -- corners ``0 1 2 3`` are ``(0,0,0) (1,0,0) (1,1,0) (0,1,0)`` and ``4 5 6 7`` the same
one node up ``z``. Edge ``e`` joins ``MC33_EDGE_CORNERS[e]``, lower corner first.

The work is parallel over lattice nodes: each node owns its three positive-axis edges and the cell
it is the lowest corner of. ``mc33_counts`` counts each node's edge vertices, triangles and cell
vertex; one inclusive scan of the counts numbers them, and ``mc33_emit`` writes them, so the output
is independent of the launch order. What a serial extraction makes order-dependent is the cell
vertex, placed at the mean of the crossings *already made* when the cell is reached; it is
reproduced from the order alone (``mc33_cell_vertex``), see there.

Tables: scikit-image's ``skimage/measure/_marching_cubes_lewiner_luts.py``, generated from
Lewiner's ``LookUpTable.h``; ``CASESCLASSIC`` is left out. One entry differs from Lewiner's
original: subcase 6.1.2 is scikit-image's corrected tiling of nine triangles around a cell vertex
where the original has seven triangles and no cell vertex. The tables are used under
scikit-image's licence:

    Copyright (C) 2009-2022 the scikit-image team. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are
    permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of
       conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of
       conditions and the following disclaimer in the documentation and/or other materials
       provided with the distribution.
    3. Neither the name of skimage nor the names of its contributors may be used to endorse or
       promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import base64

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

# Each table as ``(shape, base64 of its int8 entries)``, as scikit-image stores it.
_LUTS = {
    "CASES": (
        (256, 2),
        """
AP8BAAEBAgABAgMAAgMFAAEDAgEDAwUBAgUFBAUJCAABBAICAwQFAgQCBgIGCQsAAwgFBQcDCQEG
EA4DDAwFGAEFAwECBAUDAwYHAAUKCQAEAwYEBgsOAQYRDAQLBgUZAggFBwUMCAEGEgwFDgcFHAYV
CwQMDwUeCgUGIAYnAgwBBgQAAwUGAAIGBgMFCw4AAwkGBQcEDAEFDgsDCQQFGgMKBgYHBQwCBhMK
AQwNBhgHBwwJDQEHCQwUBiEHDQMMAgoGBwUNCwIFEAwHCAMFHQYWCgIMEQYbDgkGIgUnAg4FFA4F
CQUFIAsKBiMFKQIQDBcGJQcOAxAGLgQGAxUBCAEHAwIEAQYBAwcHAQYKDAACBwUGBgwLAQUPCQIO
BgUbAgkFCAYNDgIGFAwGCgMGGQUSCAIMEAUfCwkFIgYoAg0DCwcCBg4MAwcGDQAMDgcIBhcMCgoE
BhwMFQcKBikDDQUVCQMLCAUhDBYHCwYqAw4OCwUkBiwCEQYvAxIEBwEJAgsGCAYPCgAFEQwICwcG
GgUTDgQMEgYdCAQFIwUoAg8FFgsFDBMGHg4KBiQGKwQECQcFJQcPAxEFLAITAxYBCgUXDAsOCAYf
CQYHDAUqAw8LCwYmBi0EBQUtAxMCFQELCAUFJgUrAhIFLgMUAhYBDAUvAhQDFwENAhcBDgEPAP8=
""",
    ),
    "TILING1": (
        (16, 3),
        """
AAgDAAEJAQIKAwsCBAcICQUECgYFBwYLBwsGCgUGCQQFBAgHAwILAQoCAAkBAAMI
""",
    ),
    "TILING2": (
        (24, 6),
        """
AQgDCQgBAAsCCAsABAMABwMECQIKAAIJAAUEAQUAAwoBCwoDAQYFAgYBBwIDBgIHCQcIBQcJBggE
CwgGCgQJBgQKCwUKBwULCwoFBwsFCgkEBgoEBgQICwYICQgHBQkHBwMCBgcCAQUGAgEGAwEKCwMK
AAQFAQAFCQoCAAkCBAADBwQDAAILCAALAQMICQEI
""",
    ),
    "TILING3_1": (
        (24, 6),
        """
AAgDAQIKCQUEAAgDAwAICwcGAQkAAgMLAAEJCAQHCQABBQoGAQIKCQUECgECBgsHCAQHAwsCAgML
CgYFBQoGBAcIBAkFBwYLBQkECwYHBgoFCAcECwMCBQYKBwQIAgsDAgEKBwsGCgIBBAUJAQAJBgoF
CQEABwQIAAkBCwMCCAADBgcLBAUJAwgAAwgACgIB
""",
    ),
    "TILING3_2": (
        (24, 12),
        """
CgMCCggDCgEACAoAAwQIAwUEAwAJBQMJBggHBgAIBgsDAAYDCwADCwkACwIBCQsBBwkEBwEJBwgA
AQcABgEKBgABCQAGCQYFBAoFBAIKBAkBAgQBBwILBwECBwYKAQcKAgcLAgQHAgMIBAIIBQsGBQML
BQoCAwUCCAYHCAoGCAQFCggFCwUGCwkFCwcECQsEBgULBQkLBAcLBAsJBwYIBgoIBQQIBQgKBgsF
CwMFAgoFAgUDCwcCBwQCCAMCCAIECwIHAgEHCgYHCgcBBQoECgIEAQkEAQQCCgEGAQAGBgAJBQYJ
BAkHCQEHAAgHAAcBAwALAAkLAQILAQsJBwgGCAAGAwsGAwYACAQDBAUDCQADCQMFAgMKAwgKAAEK
AAoI
""",
    ),
    "TILING4_1": (
        (8, 6),
        """
AAgDBQoGAAEJCwcGAQIKCAQHCQUEAgMLBAUJCwMCCgIBBwQICQEABgcLAwgABgoF
""",
    ),
    "TILING4_2": (
        (8, 18),
        """
CAUABQgGAwYIBgMKAAoDCgAFCQYBBgkHAAcJBwALAQsACwEGCgcCBwoEAQQKBAEIAggBCAIHCwQD
BAsFAgULBQIJAwkCCQMEAwQLBQsECwUCCQIFAgkDBAMJAgcKBAoHCgQBCAEEAQgCBwIIAQYJBwkG
CQcACwAHAAsBBgELAAUIBggFCAYDCgMGAwoABQAK
""",
    ),
    "TILING5": (
        (48, 9),
        """
AggDAgoICgkIAQsCAQkLCQgLBAEJBAcBBwMBCAUECAMFAwEFAAoBAAgKCAsKCwQHCwIEAgAEBwAI
BwYABgIACQMACQUDBQcDAwYLAwAGAAQGAwkAAwsJCwoJBQIKBQQCBAACCQYFCQAGAAIGAAcIAAEH
AQUHCgABCgYABgQABgMLBgUDBQEDCgcGCgEHAQMHAQQJAQIEAgYECwECCwcBBwUBCAIDCAQCBAYC
AgUKAgMFAwcFBwoGBwgKCAkKBgkFBgsJCwgJBQgEBQoICgsIBAsHBAkLCQoLBAcLBAsJCQsKBQQI
BQgKCggLBgUJBgkLCwkIBwYKBwoICAoJAgoFAgUDAwUHCAMCCAIEBAIGCwIBCwEHBwEFAQkEAQQC
AgQGCgYHCgcBAQcDBgsDBgMFBQMBCgEACgAGBgAEAAgHAAcBAQcFCQUGCQYAAAYCBQoCBQIEBAIA
AwAJAwkLCwkKAwsGAwYAAAYECQADCQMFBQMHBwgABwAGBgACCwcECwQCAgQAAAEKAAoICAoLCAQF
CAUDAwUBBAkBBAEHBwEDAQILAQsJCQsIAgMIAggKCggJ
""",
    ),
    "TILING6_1_1": (
        (48, 9),
        """
BgUKAwEICQgBCwcGCQMBAwkIAQIKBwAEAAcDAwAIBQIGAgUBBQQJAgALCAsACgYFCAIAAggLCgYF
AAQDBwMEAwAIBgQKCQoECAMACgcFBwoLCAQHCgACAAoJBwYLAAIJCgkCAgMLBAEFAQQAAAEJBgMH
AwYCCQABCwQGBAsICwcGAQUABAAFAAEJBwULCgsFBAcIAQMKCwoDCQUECwEDAQsKCgECCAUHBQgJ
CAQHAgYBBQEGAQIKBAYICwgGAgMLBQcJCAkHCwIDCQYEBgkKCQUEAwcCBgIHBAUJAgcDBwIGAwIL
BAYJCgkGCwMCCQcFBwkICgIBCAYEBggLBwQIAQYCBgEFAgEKBwUICQgFBAUJAwELCgsBCAcECgMB
AwoLCQEACwUHBQsKBgcLAAUBBQAEAQAJBgQLCAsECQEABwMGAgYDCwMCBQEEAAQBCwYHCQIAAgkK
BwQIAgAKCQoAAAMIBQcKCwoHCAADCgQGBAoJBQYKAwQABAMHBQYKAAIICwgCCQQFCwACAAsICAAD
BgIFAQUCCgIBBAAHAwcABgcLAQMJCAkDCgUGCAEDAQgJ
""",
    ),
    "TILING6_1_2": (
        (48, 27),
        """
AQwDDAoDBgMKAwYIBQgGCAUMDAkIAQkMDAUKAQwDAQsMCwEGCQYBBgkHDAcJCQgMDAgDCwcMBAwA
BAEMAQQKBwoECgcCDAIHBwMMDAMAAQIMBgwCBgMMAwYIBQgGCAUADAAFBQEMDAECAwAMAAwCDAkC
BQIJAgULBAsFCwQMDAgLAAgMDAQJAAwCAAoMCgAFCAUABQgGDAYICAsMDAsCCgYMBAwADAUACgAF
AAoDBgMKAwYMDAcDBAcMDAYFBAwGDAgGAwYIBgMKAAoDCgAMDAkKBAkMDAAIBQwHBQgMCAUACgAF
AAoDDAMKCgsMDAsHCAMMAgwAAggMCAIHCgcCBwoEDAQKCgkMDAkACAQMAgwADAsABwALAAcJBgkH
CQYMDAoJAgoMDAYLBQwBBQIMAgULBAsFCwQDDAMEBAAMDAABAgMMBwwDBwAMAAcJBgkHCQYBDAEG
BgIMDAIDAAEMBgwEBgkMCQYBCwEGAQsADAALCwgMDAgECQAMBQwBDAYBCwEGAQsABwALAAcMDAQA
BQQMDAcGBQwHDAkHAAcJBwALAQsACwEMDAoLBQoMDAEJAwwBDAgBBAEIAQQKBwoECgcMDAsKAwsM
DAcIAwwBAwkMCQMECwQDBAsFDAULCwoMDAoBCQUMBwwFBwoMCgcCCAIHAggBDAEICAkMDAkFCgEM
BgwCDAcCCAIHAggBBAEIAQQMDAUBBgUMDAQHBgwEDAoEAQQKBAEIAggBCAIMDAsIBgsMDAIKBwwF
DAsFAgULBQIJAwkCCQMMDAgJBwgMDAMLBAwGBAsMCwQDCQMEAwkCDAIJCQoMDAoGCwIMBwwDDAQD
CQMEAwkCBQIJAgUMDAYCBwYMDAUEAwwHAwQMBAMJAgkDCQIFDAUCAgYMDAYHBAUMBgwEDAsEAwQL
BAMJAgkDCQIMDAoJBgoMDAILBQwHBQsMCwUCCQIFAgkDDAMJCQgMDAgHCwMMBAwGBAoMCgQBCAEE
AQgCDAIICAsMDAsGCgIMAgwGAgcMBwIIAQgCCAEEDAQBAQUMDAUGBwQMBQwHDAoHAgcKBwIIAQgC
CAEMDAkIBQkMDAEKAQwDDAkDBAMJAwQLBQsECwUMDAoLAQoMDAUJAQwDAQgMCAEECgQBBAoHDAcK
CgsMDAsDCAcMBwwFBwkMCQcACwAHAAsBDAELCwoMDAoFCQEMAQwFAQYMBgELAAsBCwAHDAcAAAQM
DAQFBgcMBAwGDAkGAQYJBgELAAsBCwAMDAgLBAgMDAAJAwwHDAAHCQcABwkGAQYJBgEMDAIGAwIM
DAEAAQwFDAIFCwUCBQsEAwQLBAMMDAAEAQAMDAMCAAwCAAsMCwAHCQcABwkGDAYJCQoMDAoCCwYM
AAwCDAgCBwIIAgcKBAoHCgQMDAkKAAkMDAQIBwwFDAgFAAUIBQAKAwoACgMMDAsKBwsMDAMIBgwE
BggMCAYDCgMGAwoADAAKCgkMDAkECAAMAAwEAAUMBQAKAwoACgMGDAYDAwcMDAcEBQYMAgwADAoA
BQAKAAUIBggFCAYMDAsIAgsMDAYKAgwAAgkMCQIFCwUCBQsEDAQLCwgMDAgACQQMAgwGDAMGCAYD
BggFAAUIBQAMDAEFAgEMDAADAAwEDAEECgQBBAoHAgcKBwIMDAMHAAMMDAIBAwwBDAsBBgELAQYJ
BwkGCQcMDAgJAwgMDAcLAwwBAwoMCgMGCAYDBggFDAUICAkMDAkBCgUM
""",
    ),
    "TILING6_2": (
        (48, 15),
        """
AQoDBgMKAwYIBQgGCAUJAQsDCwEGCQYBBgkHCAcJBAEAAQQKBwoECgcCAwIHBgMCAwYIBQgGCAUA
AQAFAAkCBQIJAgULBAsFCwQIAAoCCgAFCAUABQgGCwYIBAUACgAFAAoDBgMKAwYHBAgGAwYIBgMK
AAoDCgAJBQgHCAUACgAFAAoDCwMKAggACAIHCgcCBwoECQQKAgsABwALAAcJBgkHCQYKBQIBAgUL
BAsFCwQDAAMEBwADAAcJBgkHCQYBAgEGBgkECQYBCwEGAQsACAALBQYBCwEGAQsABwALAAcEBQkH
AAcJBwALAQsACwEKAwgBBAEIAQQKBwoECgcLAwkBCQMECwQDBAsFCgULBwoFCgcCCAIHAggBCQEI
BgcCCAIHAggBBAEIAQQFBgoEAQQKBAEIAggBCAILBwsFAgULBQIJAwkCCQMIBAsGCwQDCQMEAwkC
CgIJBwQDCQMEAwkCBQIJAgUGAwQHBAMJAgkDCQIFBgUCBgsEAwQLBAMJAgkDCQIKBQsHCwUCCQIF
AgkDCAMJBAoGCgQBCAEEAQgCCwIIAgcGBwIIAQgCCAEEBQQBBQoHAgcKBwIIAQgCCAEJAQkDBAMJ
AwQLBQsECwUKAQgDCAEECgQBBAoHCwcKBwkFCQcACwAHAAsBCgELAQYFBgELAAsBCwAHBAcABAkG
AQYJBgELAAsBCwAIAwAHCQcABwkGAQYJBgECAQIFCwUCBQsEAwQLBAMAAAsCCwAHCQcABwkGCgYJ
AAgCBwIIAgcKBAoHCgQJBwgFAAUIBQAKAwoACgMLBggECAYDCgMGAwoACQAKAAUEBQAKAwoACgMG
BwYDAgoABQAKAAUIBggFCAYLAgkACQIFCwUCBQsECAQLAgMGCAYDBggFAAUIBQABAAEECgQBBAoH
AgcKBwIDAwsBBgELAQYJBwkGCQcIAwoBCgMGCAYDBggFCQUI
""",
    ),
    "TILING7_1": (
        (16, 9),
        """
CQUECgECCAMACwcGCAMACgECAwAIBQQJBwYLCAQHCQABCwIDCgYFCwIDCQABAAEJBgUKBAcIAQIK
BwYLBQQJAgMLBAcIBgUKCwMCCAcECgUGCgIBCwYHCQQFCQEACgUGCAcEBQYKAwILAQAJBwQIAQAJ
AwILCAADCQQFCwYHBgcLAAMIAgEKBAUJAgEKAAMI
""",
    ),
    "TILING7_2": (
        (16, 3, 15),
        """
AQIKAwQIBAMFAAUDBQAJAwAICQEEAgQBBAIFCgUCCQUEAAoBCgAICggCAwIIAwAIAQYKBgEHAgcB
BwILAQIKCwMGAAYDBgAHCAcACwcGAggDCAIKCAoAAQAKCQUECwMGAAYDBgAHCAcACwcGAwQIBAMF
AAUDBQAJAwAIBAkHCwcJBQsJCwUGAAEJAgcLBwIEAwQCBAMIAgMLCAAHAQcABwEECQQBCAQHAwkA
CQMLCQsBAgELAgMLAAUJBQAGAQYABgEKAAEJCgIFAwUCBQMGCwYDBgUKAQsCCwEJCwkDAAMJBgUK
CAAHAQcABwEECQQBCAQHAAUJBQAGAQYABgEKAAEJBQoECAQKBggKCAYHCwcGCQEEAgQBBAIFCgUC
CQUEAQYKBgEHAgcBBwILAQIKBgsFCQULBwkLCQcECAQHCgIFAwUCBQMGCwYDBgUKAgcLBwIEAwQC
BAMIAgMLBwgGCgYIBAoICgQFBwQIBQIKAgUDBgMFAwYLCgUGCwcCBAIHAgQDCAMECwMCBggHCAYK
CAoEBQQKBgcLBAEJAQQCBQIEAgUKBAUJCgYBBwEGAQcCCwIHCgIBBQsGCwUJCwkHBAcJCgUGBwAI
AAcBBAEHAQQJBwQICQUABgAFAAYBCgEGCQEABAoFCgQICggGBwYICwMCCQUABgAFAAYBCgEGCQEA
BQIKAgUDBgMFAwYLCgUGAgsBCQELAwkLCQMACQEACwcCBAIHAgQDCAMECwMCBwAIAAcBBAEHAQQJ
BwQIAAkDCwMJAQsJCwECBAUJBgMLAwYABwAGAAcIBgcLCAQDBQMEAwUACQAFCAADBwkECQcLCQsF
BgULCAADCgYBBwEGAQcCCwIHCgIBBgMLAwYABwAGAAcIBgcLAwgCCgIIAAoICgABCgIBCAQDBQME
AwUACQAFCAADBAEJAQQCBQIEAgUKBAUJAQoACAAKAggKCAID
""",
    ),
    "TILING7_3": (
        (16, 3, 27),
        """
DAIKDAoFDAUEDAQIDAgDDAMADAAJDAkBDAECDAUEDAQIDAgDDAMCDAIKDAoBDAEADAAJDAkFBQQM
CgUMAgoMAwIMCAMMAAgMAQAMCQEMBAkMDAAIDAgHDAcGDAYKDAoBDAECDAILDAsDDAMADAcGDAYK
DAoBDAEADAAIDAgDDAMCDAILDAsHBwYMCAcMAAgMAQAMCgEMAgoMAwIMCwMMBgsMCQUMAAkMAwAM
CwMMBgsMBwYMCAcMBAgMBQQMAwAMCwMMBgsMBQYMCQUMBAkMBwQMCAcMAAgMDAMADAAJDAkFDAUG
DAYLDAsHDAcEDAQIDAgDDAEJDAkEDAQHDAcLDAsCDAIDDAMIDAgADAABDAQHDAcLDAsCDAIBDAEJ
DAkADAADDAMIDAgEBAcMCQQMAQkMAgEMCwIMAwsMAAMMCAAMBwgMDAMLDAsGDAYFDAUJDAkADAAB
DAEKDAoCDAIDDAYFDAUJDAkADAADDAMLDAsCDAIBDAEKDAoGBgUMCwYMAwsMAAMMCQAMAQkMAgEM
CgIMBQoMCgYMAQoMAAEMCAAMBwgMBAcMCQQMBQkMBgUMAAEMCAAMBwgMBgcMCgYMBQoMBAUMCQQM
AQkMDAABDAEKDAoGDAYHDAcIDAgEDAQFDAUJDAkACwcMAgsMAQIMCQEMBAkMBQQMCgUMBgoMBwYM
AQIMCQEMBAkMBwQMCwcMBgsMBQYMCgUMAgoMDAECDAILDAsHDAcEDAQJDAkFDAUGDAYKDAoBCAQM
AwgMAgMMCgIMBQoMBgUMCwYMBwsMBAcMAgMMCgIMBQoMBAUMCAQMBwgMBgcMCwYMAwsMDAIDDAMI
DAgEDAQFDAUKDAoGDAYHDAcLDAsCDAQIDAgDDAMCDAIKDAoFDAUGDAYLDAsHDAcEDAMCDAIKDAoF
DAUEDAQIDAgHDAcGDAYLDAsDAwIMCAMMBAgMBQQMCgUMBgoMBwYMCwcMAgsMDAcLDAsCDAIBDAEJ
DAkEDAQFDAUKDAoGDAYHDAIBDAEJDAkEDAQHDAcLDAsGDAYFDAUKDAoCAgEMCwIMBwsMBAcMCQQM
BQkMBgUMCgYMAQoMDAYKDAoBDAEADAAIDAgHDAcEDAQJDAkFDAUGDAEADAAIDAgHDAcGDAYKDAoF
DAUEDAQJDAkBAQAMCgEMBgoMBwYMCAcMBAgMBQQMCQUMAAkMCwMMBgsMBQYMCQUMAAkMAQAMCgEM
AgoMAwIMBQYMCQUMAAkMAwAMCwMMAgsMAQIMCgEMBgoMDAUGDAYLDAsDDAMADAAJDAkBDAECDAIK
DAoFCQEMBAkMBwQMCwcMAgsMAwIMCAMMAAgMAQAMBwQMCwcMAgsMAQIMCQEMAAkMAwAMCAMMBAgM
DAcEDAQJDAkBDAECDAILDAsDDAMADAAIDAgHDAUJDAkADAADDAMLDAsGDAYHDAcIDAgEDAQFDAAD
DAMLDAsGDAYFDAUJDAkEDAQHDAcIDAgAAAMMCQAMBQkMBgUMCwYMBwsMBAcMCAQMAwgMCAAMBwgM
BgcMCgYMAQoMAgEMCwIMAwsMAAMMBgcMCgYMAQoMAAEMCAAMAwgMAgMMCwIMBwsMDAYHDAcIDAgA
DAABDAEKDAoCDAIDDAMLDAsGCgIMBQoMBAUMCAQMAwgMAAMMCQAMAQkMAgEMBAUMCAQMAwgMAgMM
CgIMAQoMAAEMCQAMBQkMDAQFDAUKDAoCDAIDDAMIDAgADAABDAEJDAkE
""",
    ),
    "TILING7_4_1": (
        (16, 15),
        """
AwQIBAMKAgoDBAoFCQEAAQYKBgEIAAgBBggHCwMCCwMGCQYDBgkFAAkDBwQIAgcLBwIJAQkCBwkE
CAADAAUJBQALAwsABQsGCgIBCAAHCgcABwoGAQoABAUJCQEECwQBBAsHAgsBBQYKCgIFCAUCBQgE
AwgCBgcLBQIKAgUIBAgFAggDCwcGBAEJAQQLBwsEAQsCCgYFBwAIAAcKBgoHAAoBCQUECQUACwAF
AAsDBgsFAQIKCwcCCQIHAgkBBAkHAwAIBgMLAwYJBQkGAwkACAQHCgYBCAEGAQgABwgGAgMLCAQD
CgMEAwoCBQoEAAEJ
""",
    ),
    "TILING7_4_2": (
        (16, 27),
        """
CQQIBAkFCgUJAQoJCgECAAIBAgADCAMACQgACwYKBgsHCAcLAwgLCAMAAgADAAIBCgECCwoCCwMI
AAgDCAAJCAkEBQQJBAUHBgcFBwYLBwsICAcLBwgECQQIAAkICQABAwEAAQMCCwIDCAsDCgUJBQoG
CwYKAgsKCwIDAQMCAwEACQABCgkBCAAJAQkACQEKCQoFBgUKBQYEBwQGBAcIBAgJCQEKAgoBCgIL
CgsGBwYLBgcFBAUHBQQJBQkKCgILAwsCCwMICwgHBAcIBwQGBQYEBgUKBgoLCwIKAgsDCAMLBwgL
CAcEBgQHBAYFCgUGCwoGCgEJAQoCCwIKBgsKCwYHBQcGBwUECQQFCgkFCQAIAAkBCgEJBQoJCgUG
BAYFBgQHCAcECQgECQUKBgoFCgYLCgsCAwILAgMBAAEDAQAJAQkKCwcIBAgHCAQJCAkAAQAJAAED
AgMBAwILAwsICAMLAwgACQAIBAkICQQFBwUEBQcGCwYHCAsHCgYLBwsGCwcICwgDAAMIAwACAQIA
AgEKAgoLCAQJBQkECQUKCQoBAgEKAQIAAwACAAMIAAgJ
""",
    ),
    "TILING8": (
        (6, 6),
        """
CQgKCggLAQUDAwUHAAQCBAYCAAIEBAIGAQMFAwcFCQoICgsI
""",
    ),
    "TILING9": (
        (8, 12),
        """
AgoFAwIFAwUEAwQIBAcLCQQLCQsCCQIBCgcGAQcKAQgHAQAIAwYLAAYDAAUGAAkFAwsGAAMGAAYF
AAUJCgYHAQoHAQcIAQgABAsHCQsECQILCQECAgUKAwUCAwQFAwgE
""",
    ),
    "TILING10_1_1": (
        (6, 12),
        """
BQoHCwcKCAEJAQgDAQIFBgUCBAMAAwQHCwAIAAsCBAkGCgYJCQAKAgoABggECAYLBwIDAgcGAAEE
BQQBBwkFCQcICgELAwsB
""",
    ),
    "TILING10_1_1_": (
        (6, 12),
        """
BQkHCAcJCwEKAQsDAwIHBgcCBAEAAQQFCgAJAAoCBAgGCwYICAALAgsABgkECQYKBQIBAgUGAAME
BwQDBwoFCgcLCQEIAwgB
""",
    ),
    "TILING10_1_2": (
        (6, 24),
        """
AwsHAwcICQgHBQkHCQUKCQoBAwEKCwMKBwYFBwUEAAQFAQAFAAECAAIDBwMCBgcCCwIKBgsKCwYE
CwQIAAgECQAEAAkKAAoCCwIKCwoGBAYKCQQKBAkABAAICwgAAgsABwYFBAcFBwQABwADAgMAAQIA
AgEFAgUGBwgDCwcDBwsKBwoFCQUKAQkKCQEDCQMI
""",
    ),
    "TILING10_2": (
        (6, 24),
        """
DAUJDAkIDAgDDAMBDAEKDAoLDAsHDAcFDAEADAAEDAQHDAcDDAMCDAIGDAYFDAUBBAgMBgQMCgYM
CQoMAAkMAgAMCwIMCAsMDAkEDAQGDAYLDAsIDAgADAACDAIKDAoJAAMMBAAMBQQMAQUMAgEMBgIM
BwYMAwcMCgUMCwoMAwsMAQMMCQEMCAkMBwgMBQcM
""",
    ),
    "TILING10_2_": (
        (6, 24),
        """
CAcMCQgMAQkMAwEMCwMMCgsMBQoMBwUMBAUMAAQMAwAMBwMMBgcMAgYMAQIMBQEMDAsGDAYEDAQJ
DAkKDAoCDAIADAAIDAgLBgoMBAYMCAQMCwgMAgsMAAIMCQAMCgkMDAcEDAQADAABDAEFDAUGDAYC
DAIDDAMHDAcLDAsKDAoBDAEDDAMIDAgJDAkFDAUH
""",
    ),
    "TILING11": (
        (12, 12),
        """
AgoJAgkHAgcDBwkEAQYCAQgGAQkICAcGCAMBCAEGCAYEBgEKAAgLAAsFAAUBBQsGCQUHCQcCCQIA
AgcLBQAEBQsABQoLCwMABQQABQALBQsKCwADCQcFCQIHCQACAgsHAAsIAAULAAEFBQYLCAEDCAYB
CAQGBgoBAQIGAQYIAQgJCAYHAgkKAgcJAgMHBwQJ
""",
    ),
    "TILING12_1_1": (
        (24, 12),
        """
BwYLCgMCAwoICQgKBgUKCQIBAgkLCAsJCgYFBwkECQcBAwEHBwYLBAgFAwUIBQMBBQQJCAEAAQgK
CwoIAQIKAAkDBQMJAwUHCgECAAsDCwAGBAYACAMAAgkBCQIEBgQCAwAIAgsBBwELAQcFBgUKBwsE
AgQLBAIACQUEBggHCAYAAgAGCAMABwQLCQsECwkKBAcICwADAAsJCgkLBAcIBQkGAAYJBgACCwcG
BAoFCgQCAAIECwIDAQgACAEHBQcBAAEJAwgCBAIIAgQGAgMLAQoABgAKAAYECQABAwoCCgMFBwUD
CQABBAUICggFCAoLCAQHBQsGCwUDAQMFBQQJBgoHAQcKBwEDCgECBQYJCwkGCQsICwIDBgcKCAoH
CggJ
""",
    ),
    "TILING12_1_1_": (
        (24, 12),
        """
AwILCgcGBwoICQgKAgEKCQYFBgkLCAsJCQQFBwoGCgcBAwEHBwQIBgsFAwULBQMBAQAJCAUEBQgK
CwoIAQAJAgoDBQMKAwUHCwMCAAoBCgAGBAYACQEAAggDCAIEBgQCAwILAAgBBwEIAQcFBgcLBQoE
AgQKBAIACAcEBgkFCQYAAgAGCAcEAwALCQsACwkKAAMICwQHBAsJCgkLBAUJBwgGAAYIBgACCgUG
BAsHCwQCAAIECAADAQsCCwEHBQcBAAMIAQkCBAIJAgQGAgEKAwsABgALAAYECgIBAwkACQMFBwUD
CQQFAAEICggBCAoLCwYHBQgECAUDAQMFBQYKBAkHAQcJBwEDCgUGAQIJCwkCCQsICwYHAgMKCAoD
CggJ
""",
    ),
    "TILING12_1_2": (
        (24, 24),
        """
BwMLAwcICQgHBgkHCQYKAgoGCwIGAgsDBgIKAgYLCAsGBQgGCAUJAQkFCgEFAQoCCgkFCQoBAwEK
BgMKAwYHBAcGBQQGBAUJBwgLAwsICwMBCwEGBQYBBgUEBgQHCAcEBQEJAQUKCwoFBAsFCwQIAAgE
CQAEAAkBAQkKBQoJCgUHCgcCAwIHAgMAAgABCQEACgsCCwoGBAYKAQQKBAEAAwABAgMBAwILCAkA
CQgEBgQIAwYIBgMCAQIDAAEDAQAJAwsIBwgLCAcFCAUAAQAFAAECAAIDCwMCBgsKAgoLCgIACgAF
BAUABQQHBQcGCwYHCQgECAkAAgAJBQIJAgUGBwYFBAcFBwQICAQACQAEAAkKAAoDCwMKAwsHAwcI
BAgHBAAIAAQJCgkEBwoECgcLAwsHCAMHAwgABAkIAAgJCAACCAIHBgcCBwYFBwUECQQFCwoGCgsC
AAILBwALAAcEBQQHBgUHBQYKCwgDCAsHBQcLAgULBQIBAAECAwACAAMIAAgJBAkICQQGCQYBAgEG
AQIDAQMACAADAgoLBgsKCwYECwQDAAMEAwABAwECCgIBCQoBCgkFBwUJAAcJBwADAgMAAQIAAgEK
CQUBCgEFAQoLAQsACAALAAgEAAQJBQkECAsHCwgDAQMIBAEIAQQFBgUEBwYEBgcLBQoJAQkKCQED
CQMEBwQDBAcGBAYFCgUGCgYCCwIGAgsIAggBCQEIAQkFAQUKBgoFCwcDCAMHAwgJAwkCCgIJAgoG
AgYLBwsG
""",
    ),
    "TILING12_2": (
        (24, 24),
        """
CQgMCgkMAgoMAwIMCwMMBgsMBwYMCAcMCAsMCQgMAQkMAgEMCgIMBQoMBgUMCwYMAwEMBwMMBAcM
CQQMBQkMBgUMCgYMAQoMDAMBDAEFDAUGDAYLDAsHDAcEDAQIDAgDCwoMCAsMAAgMAQAMCQEMBAkM
BQQMCgUMDAUHDAcDDAMCDAIKDAoBDAEADAAJDAkFBAYMAAQMAQAMCgEMAgoMAwIMCwMMBgsMBgQM
AgYMAwIMCAMMAAgMAQAMCQEMBAkMDAcFDAUBDAEADAAIDAgDDAMCDAILDAsHDAIADAAEDAQFDAUK
DAoGDAYHDAcLDAsCAgAMBgIMBwYMCAcMBAgMBQQMCQUMAAkMDAkKDAoLDAsHDAcEDAQIDAgDDAMA
DAAJCgkMCwoMBwsMBAcMCAQMAwgMAAMMCQAMDAACDAIGDAYHDAcIDAgEDAQFDAUJDAkAAAIMBAAM
BQQMCgUMBgoMBwYMCwcMAgsMBQcMAQUMAAEMCAAMAwgMAgMMCwIMBwsMDAQGDAYCDAIDDAMIDAgA
DAABDAEJDAkEDAYEDAQADAABDAEKDAoCDAIDDAMLDAsGBwUMAwcMAgMMCgIMAQoMAAEMCQAMBQkM
DAoLDAsIDAgADAABDAEJDAkEDAQFDAUKAQMMBQEMBgUMCwYMBwsMBAcMCAQMAwgMDAEDDAMHDAcE
DAQJDAkFDAUGDAYKDAoBDAsIDAgJDAkBDAECDAIKDAoFDAUGDAYLDAgJDAkKDAoCDAIDDAMLDAsG
DAYHDAcI
""",
    ),
    "TILING12_2_": (
        (24, 24),
        """
DAILDAsHDAcGDAYKDAoJDAkIDAgDDAMCDAEKDAoGDAYFDAUJDAkIDAgLDAsCDAIBDAQFDAUKDAoG
DAYHDAcDDAMBDAEJDAkEBwYMCAcMBAgMBQQMAQUMAwEMCwMMBgsMDAAJDAkFDAUEDAQIDAgLDAsK
DAoBDAEAAQIMCQEMAAkMAwAMBwMMBQcMCgUMAgoMDAECDAILDAsDDAMADAAEDAQGDAYKDAoBDAMA
DAAJDAkBDAECDAIGDAYEDAQIDAgDAwAMCwMMAgsMAQIMBQEMBwUMCAcMAAgMBgUMCwYMBwsMBAcM
AAQMAgAMCgIMBQoMDAcEDAQJDAkFDAUGDAYCDAIADAAIDAgHCAcMAAgMAwAMCwMMCgsMCQoMBAkM
BwQMDAcIDAgADAADDAMLDAsKDAoJDAkEDAQHBAcMCQQMBQkMBgUMAgYMAAIMCAAMBwgMDAUGDAYL
DAsHDAcEDAQADAACDAIKDAoFDAADDAMLDAsCDAIBDAEFDAUHDAcIDAgAAAMMCQAMAQkMAgEMBgIM
BAYMCAQMAwgMAgEMCwIMAwsMAAMMBAAMBgQMCgYMAQoMDAIBDAEJDAkADAADDAMHDAcFDAUKDAoC
CQAMBQkMBAUMCAQMCwgMCgsMAQoMAAEMDAYHDAcIDAgEDAQFDAUBDAEDDAMLDAsGBQQMCgUMBgoM
BwYMAwcMAQMMCQEMBAkMCgEMBgoMBQYMCQUMCAkMCwgMAgsMAQIMCwIMBwsMBgcMCgYMCQoMCAkM
AwgMAgMM
""",
    ),
    "TILING13_1": (
        (2, 12),
        """
CwcGAQIKCAMACQUECAQHAgMLCQABCgYF
""",
    ),
    "TILING13_1_": (
        (2, 12),
        """
BwQICwMCAQAJBQYKBgcLCgIBAAMIBAUJ
""",
    ),
    "TILING13_2": (
        (2, 6, 18),
        """
AQIKCwcGAwQIBAMFAAUDBQAJCAMACwcGCQEEAgQBBAIFCgUCCQUECAMAAQYKBgEHAgcBBwILCQUE
AQIKCwMGAAYDBgAHCAcACQUECwcGAAoBCgAICggCAwIIAQIKAwAIBAkHCwcJBQsJCwUGAgMLCAQH
AAUJBQAGAQYABgEKCQABCAQHCgIFAwUCBQMGCwYDBgUKCQABAgcLBwIEAwQCBAMIBgUKAgMLCAAH
AQcABwEECQQBBgUKCAQHAQsCCwEJCwkDAAMJAgMLAAEJBQoECAQKBggKCAYH
""",
    ),
    "TILING13_2_": (
        (2, 6, 18),
        """
CgUGCwMCBwAIAAcBBAEHAQQJCwMCBwQICQUABgAFAAYBCgEGAQAJBwQIBQIKAgUDBgMFAwYLCgUG
AQAJCwcCBAIHAgQDCAMECgUGBwQIAgsBCQELAwkLCQMACwMCCQEABAoFCgQICggGBwYIBgcLCAAD
BAEJAQQCBQIEAgUKCAADBAUJCgYBBwEGAQcCCwIHAgEKBAUJBgMLAwYABwAGAAcIBgcLAgEKCAQD
BQMEAwUACQAFBgcLBAUJAwgCCgIIAAoICgABCAADCgIBBQsGCwUJCwkHBAcJ
""",
    ),
    "TILING13_3": (
        (2, 12, 30),
        """
CwcGDAIKDAoFDAUEDAQIDAgDDAMADAAJDAkBDAECAQIKCQUMAAkMAwAMCwMMBgsMBwYMCAcMBAgM
BQQMCwcGDAUEDAQIDAgDDAMCDAIKDAoBDAEADAAJDAkFAQIKDAMADAAJDAkFDAUGDAYLDAsHDAcE
DAQIDAgDCAMACwcMAgsMAQIMCQEMBAkMBQQMCgUMBgoMBwYMCwcGBQQMCgUMAgoMAwIMCAMMAAgM
AQAMCQEMBAkMCAMAAQIMCQEMBAkMBwQMCwcMBgsMBQYMCgUMAgoMCQUEDAAIDAgHDAcGDAYKDAoB
DAECDAILDAsDDAMACQUEDAcGDAYKDAoBDAEADAAIDAgDDAMCDAILDAsHCAMADAECDAILDAsHDAcE
DAQJDAkFDAUGDAYKDAoBCQUEBwYMCAcMAAgMAQAMCgEMAgoMAwIMCwMMBgsMAQIKAwAMCwMMBgsM
BQYMCQUMBAkMBwQMCAcMAAgMCAQHDAMLDAsGDAYFDAUJDAkADAABDAEKDAoCDAIDAgMLCgYMAQoM
AAEMCAAMBwgMBAcMCQQMBQkMBgUMCAQHDAYFDAUJDAkADAADDAMLDAsCDAIBDAEKDAoGAgMLDAAB
DAEKDAoGDAYHDAcIDAgEDAQFDAUJDAkAAAEJCAQMAwgMAgMMCgIMBQoMBgUMCwYMBwsMBAcMCAQH
BgUMCwYMAwsMAAMMCQAMAQkMAgEMCgIMBQoMCQABAgMMCgIMBQoMBAUMCAQMBwgMBgcMCwYMAwsM
BgUKDAEJDAkEDAQHDAcLDAsCDAIDDAMIDAgADAABBgUKDAQHDAcLDAsCDAIBDAEJDAkADAADDAMI
DAgECQABDAIDDAMIDAgEDAQFDAUKDAoGDAYHDAcLDAsCBgUKBAcMCQQMAQkMAgEMCwIMAwsMAAMM
CAAMBwgMAgMLAAEMCAAMBwgMBgcMCgYMBQoMBAUMCQQMAQkM
""",
    ),
    "TILING13_3_": (
        (2, 12, 30),
        """
AwILCAcMAAgMAQAMCgEMBgoMBQYMCQUMBAkMBwQMBQYKDAILDAsHDAcEDAQJDAkBDAEADAAIDAgD
DAMCCgUGDAcEDAQJDAkBDAECDAILDAsDDAMADAAIDAgHCwMCDAEADAAIDAgHDAcGDAYKDAoFDAUE
DAQJDAkBBwQICwMMBgsMBQYMCQUMAAkMAQAMCgEMAgoMAwIMBwQIBQYMCQUMAAkMAwAMCwMMAgsM
AQIMCgEMBgoMCwMCAQAMCgEMBgoMBwYMCAcMBAgMBQQMCQUMAAkMAQAJDAQIDAgDDAMCDAIKDAoF
DAUGDAYLDAsHDAcEBwQIDAUGDAYLDAsDDAMADAAJDAkBDAECDAIKDAoFAQAJDAMCDAIKDAoFDAUE
DAQIDAgHDAcGDAYLDAsDCgUGBwQMCwcMAgsMAQIMCQEMAAkMAwAMCAMMBAgMCQEAAwIMCAMMBAgM
BQQMCgUMBgoMBwYMCwcMAgsMAAMICQQMAQkMAgEMCwIMBwsMBgcMCgYMBQoMBAUMCwYHDAMIDAgE
DAQFDAUKDAoCDAIBDAEJDAkADAADBgcLDAQFDAUKDAoCDAIDDAMIDAgADAABDAEJDAkECAADDAIB
DAEJDAkEDAQHDAcLDAsGDAYFDAUKDAoCBAUJCAAMBwgMBgcMCgYMAQoMAgEMCwIMAwsMAAMMBAUJ
BgcMCgYMAQoMAAEMCAAMAwgMAgMMCwIMBwsMCAADAgEMCwIMBwsMBAcMCQQMBQkMBgUMCgYMAQoM
AgEKDAUJDAkADAADDAMLDAsGDAYHDAcIDAgEDAQFBAUJDAYHDAcIDAgADAABDAEKDAoCDAIDDAML
DAsGAgEKDAADDAMLDAsGDAYFDAUJDAkEDAQHDAcIDAgABgcLBAUMCAQMAwgMAgMMCgIMAQoMAAEM
CQAMBQkMCgIBAAMMCQAMBQkMBgUMCwYMBwsMBAcMCAQMAwgM
""",
    ),
    "TILING13_4": (
        (2, 4, 36),
        """
DAIKDAoFDAUGDAYLDAsHDAcEDAQIDAgDDAMADAAJDAkBDAECCwMMBgsMBwYMCAcMBAgMBQQMCQUM
AAkMAQAMCgEMAgoMAwIMCQEMBAkMBQQMCgUMBgoMBwYMCwcMAgsMAwIMCAMMAAgMAQAMDAAIDAgH
DAcEDAQJDAkFDAUGDAYKDAoBDAECDAILDAsDDAMADAMLDAsGDAYHDAcIDAgEDAQFDAUJDAkADAAB
DAEKDAoCDAIDCAAMBwgMBAcMCQQMBQkMBgUMCgYMAQoMAgEMCwIMAwsMAAMMCgIMBQoMBgUMCwYM
BwsMBAcMCAQMAwgMAAMMCQAMAQkMAgEMDAEJDAkEDAQFDAUKDAoGDAYHDAcLDAsCDAIDDAMIDAgA
DAAB
""",
    ),
    "TILING13_5_1": (
        (2, 4, 18),
        """
BwYLAQAJCgMCAwoFAwUIBAgFAQIKBwQIAwALBgsACQYABgkFAwAIBQYKAQIJBAkCCwQCBAsHBQQJ
AwILCAEAAQgHAQcKBgoHBAcIAgEKCwADAAsGAAYJBQkGAgMLBAUJAAEIBwgBCgcBBwoGAAEJBgcL
AgMKBQoDCAUDBQgEBgUKAAMICQIBAgkEAgQLBwsE
""",
    ),
    "TILING13_5_2": (
        (2, 4, 30),
        """
AQAJBwQIBwgDBwMLAgsDCwIKCwoGBQYKBgUHBAcFBwQICwMCBgsCCgYCBgoFCQUKAQkKCQEAAgAB
AAIDBQYKCQEABAkACAQABAgHCwcIAwsICwMCAAIDAgABAwILBQYKBQoBBQEJAAkBCQAICQgEBAgH
BAcFBgUHAgEKBAUJBAkABAAIAwgACAMLCAsHBgcLBwYEBQQGBAUJCAADBwgDCwcDBwsGCgYLAgoL
CgIBAwECAQMABgcLCgIBBQoBCQUBBQkECAQJAAgJCAADAQMAAwECAAMIBgcLBgsCBgIKAQoCCgEJ
CgkFBQkEBQQGBwYE
""",
    ),
    "TILING14": (
        (12, 12),
        """
BQkIBQgCBQIGAwIIAgEFAgUIAggLBAgFCQQGCQYDCQMBCwMGAQsKAQQLAQAEBwsECAIACAUCCAcF
CgIFAAcDAAoHAAkKBgcKAAMHAAcKAAoJBgoHCAACCAIFCAUHCgUCAQoLAQsEAQQABwQLCQYECQMG
CQEDCwYDAgUBAggFAgsIBAUIBQgJBQIIBQYCAwgC
""",
    ),
    "TEST3": (
        (24,),
        """
BQEEBQECAgMEAwYG+vr9/P3+/v/7/P/7
""",
    ),
    "TEST4": (
        (8,),
        """
BwcHB/n5+fk=
""",
    ),
    "TEST6": (
        (48, 3),
        """
AgcKBAcLBQcBBQcDAQcJAwcKBgcFAQcIBAcIAQcIAwcLBQcCBQcAAQcJBgcGAgcJBAcIAgcJAgcK
BgcHAwcKBAcLAwcLBgcE+vkE/fkL/PkL/fkK+vkH/vkK/vkJ/PkI/vkJ+vkG//kJ+/kA+/kC/fkL
//kI/PkI//kI+vkF/fkK//kJ+/kD+/kB/PkL/vkK
""",
    ),
    "TEST7": (
        (16, 5),
        """
AQIFBwEDBAUHAwQBBgcEBAEFBwACAwUHAgECBgcFAgMGBwYDBAYHB/38+vkH/v36+Qb//vr5Bf79
+/kC/P/7+QD8//r5BP38+/kD//77+QE=
""",
    ),
    "TEST10": (
        (6, 3),
        """
AgQHBQYHAQMHAQMHBQYHAgQH
""",
    ),
    "TEST12": (
        (24, 4),
        """
BAMHCwMCBwoCBgcFBgQHBwIBBwkFAgcBBQMHAgUBBwAFBAcDBgMHBgEGBwQBBAcIBAEHCAYBBwQD
BgcGBAUHAwEFBwADBQcCAgUHAQECBwkEBgcHBgIHBQIDBwoDBAcL
""",
    ),
    "TEST13": (
        (2, 7),
        """
AQIDBAUGBwIDBAEFBgc=
""",
    ),
    "SUBCONFIG13": (
        (64,),
        """
AAECBwP/C/8ECP//Dv///wUJDBcP/xUmERT/JBohHiwGCg0TEP8ZJRIY/yMWIB0r////Iv//HCr/
H/8pGygnLQ==
""",
    ),
}


def _table(name: str) -> np.ndarray:
    """One of ``_LUTS``' tables, decoded."""
    shape, text = _LUTS[name]
    return np.frombuffer(base64.decodebytes(text.encode()), dtype=np.int8).reshape(shape)


# The four corners of each cell face, in the order the face test reads them (``A B C D`` around the
# face), for face codes ``1``-``6``; a code's sign says which side the test favours.
_FACE_CORNERS = ((0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0), (0, 3, 2, 1), (4, 7, 6, 5))

# The interior test along a reference edge: the edge's two corners, then the three edges parallel
# to it, each as the pair the test interpolates at the reference edge's crossing parameter. The
# four interpolated values are the field on the plane through the crossing, at the corners of the
# section of the cell it cuts.
_INTERIOR_EDGES = (
    (0, 1, 3, 2, 7, 6, 4, 5),
    (1, 2, 0, 3, 4, 7, 5, 6),
    (2, 3, 1, 0, 5, 4, 6, 7),
    (3, 0, 2, 1, 6, 5, 7, 4),
    (4, 5, 7, 6, 3, 2, 0, 1),
    (5, 6, 4, 7, 0, 3, 1, 2),
    (6, 7, 5, 4, 1, 0, 2, 3),
    (7, 4, 6, 5, 2, 1, 3, 0),
    (0, 4, 3, 7, 2, 6, 1, 5),
    (1, 5, 0, 4, 3, 7, 2, 6),
    (2, 6, 1, 5, 0, 4, 3, 7),
    (3, 7, 2, 6, 1, 5, 0, 4),
)

# Each edge's two corners, lower corner first: four ``x`` / ``y`` edges round the ``z = 0`` face,
# the same four round ``z = 1``, then the four ``z`` edges.
MC33_EDGE_CORNERS = (
    (0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7), (0, 4), (1, 5), (2, 6), (3, 7)
)  # fmt: skip

_TILINGS = (
    "TILING1", "TILING2", "TILING3_1", "TILING3_2", "TILING4_1", "TILING4_2", "TILING5",
    "TILING6_1_1", "TILING6_1_2", "TILING6_2", "TILING7_1", "TILING7_2", "TILING7_3",
    "TILING7_4_1", "TILING7_4_2", "TILING8", "TILING9", "TILING10_1_1", "TILING10_1_1_",
    "TILING10_1_2", "TILING10_2", "TILING10_2_", "TILING11", "TILING12_1_1", "TILING12_1_1_",
    "TILING12_1_2", "TILING12_2", "TILING12_2_", "TILING13_1", "TILING13_1_", "TILING13_2",
    "TILING13_2_", "TILING13_3", "TILING13_3_", "TILING13_4", "TILING13_5_1", "TILING13_5_2",
    "TILING14",
)  # fmt: skip


def _pack() -> tuple[tuple[int, ...], dict[str, tuple[int, int, int]]]:
    """
    Every table as one flat ``int32`` tuple, and each section's ``(base, row, width)``.

    ``row`` is a configuration's stride and ``width`` a subcase's (equal for a table without
    subcases), so a section's entry is ``base + config * row + subcase * width``.
    """
    sections: dict[str, tuple[int, int, int]] = {}
    flat: list[int] = []

    def add(name: str, array: np.ndarray) -> None:
        width = int(array.shape[-1]) if array.ndim > 1 else 1
        row = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
        sections[name] = (len(flat), row, width)
        flat.extend(int(x) for x in array.reshape(-1))

    for name in ("CASES", "TEST3", "TEST4", "TEST6", "TEST7", "TEST10", "TEST12", "TEST13"):
        add(name, _table(name))
    add("SUBCONFIG13", _table("SUBCONFIG13"))
    add("FACE_CORNERS", np.asarray(_FACE_CORNERS))
    add("INTERIOR_EDGES", np.asarray(_INTERIOR_EDGES))
    add("EDGE_CORNERS", np.asarray(MC33_EDGE_CORNERS))
    for name in _TILINGS:
        add(name, _table(name))
    return tuple(flat), sections


MC33_TABLE, _SECTIONS = _pack()

# Section bases and strides, as plain ints the kernels read as constants.
_CASES = _SECTIONS["CASES"][0]
_TEST3 = _SECTIONS["TEST3"][0]
_TEST4 = _SECTIONS["TEST4"][0]
_TEST6 = _SECTIONS["TEST6"][0]
_TEST7 = _SECTIONS["TEST7"][0]
_TEST10 = _SECTIONS["TEST10"][0]
_TEST12 = _SECTIONS["TEST12"][0]
_TEST13 = _SECTIONS["TEST13"][0]
_SUBCONFIG13 = _SECTIONS["SUBCONFIG13"][0]
_FACE = _SECTIONS["FACE_CORNERS"][0]
_INTERIOR = _SECTIONS["INTERIOR_EDGES"][0]
_EDGES = _SECTIONS["EDGE_CORNERS"][0]
_T1, _T1_ROW, _T1_WIDTH = _SECTIONS["TILING1"]
_T2, _T2_ROW, _T2_WIDTH = _SECTIONS["TILING2"]
_T3_1, _T3_1_ROW, _T3_1_WIDTH = _SECTIONS["TILING3_1"]
_T3_2, _T3_2_ROW, _T3_2_WIDTH = _SECTIONS["TILING3_2"]
_T4_1, _T4_1_ROW, _T4_1_WIDTH = _SECTIONS["TILING4_1"]
_T4_2, _T4_2_ROW, _T4_2_WIDTH = _SECTIONS["TILING4_2"]
_T5, _T5_ROW, _T5_WIDTH = _SECTIONS["TILING5"]
_T6_1_1, _T6_1_1_ROW, _T6_1_1_WIDTH = _SECTIONS["TILING6_1_1"]
_T6_1_2, _T6_1_2_ROW, _T6_1_2_WIDTH = _SECTIONS["TILING6_1_2"]
_T6_2, _T6_2_ROW, _T6_2_WIDTH = _SECTIONS["TILING6_2"]
_T7_1, _T7_1_ROW, _T7_1_WIDTH = _SECTIONS["TILING7_1"]
_T7_2, _T7_2_ROW, _T7_2_WIDTH = _SECTIONS["TILING7_2"]
_T7_3, _T7_3_ROW, _T7_3_WIDTH = _SECTIONS["TILING7_3"]
_T7_4_1, _T7_4_1_ROW, _T7_4_1_WIDTH = _SECTIONS["TILING7_4_1"]
_T7_4_2, _T7_4_2_ROW, _T7_4_2_WIDTH = _SECTIONS["TILING7_4_2"]
_T8, _T8_ROW, _T8_WIDTH = _SECTIONS["TILING8"]
_T9, _T9_ROW, _T9_WIDTH = _SECTIONS["TILING9"]
_T10_1_1, _T10_1_1_ROW, _T10_1_1_WIDTH = _SECTIONS["TILING10_1_1"]
_T10_1_1_, _T10_1_1__ROW, _T10_1_1__WIDTH = _SECTIONS["TILING10_1_1_"]
_T10_1_2, _T10_1_2_ROW, _T10_1_2_WIDTH = _SECTIONS["TILING10_1_2"]
_T10_2, _T10_2_ROW, _T10_2_WIDTH = _SECTIONS["TILING10_2"]
_T10_2_, _T10_2__ROW, _T10_2__WIDTH = _SECTIONS["TILING10_2_"]
_T11, _T11_ROW, _T11_WIDTH = _SECTIONS["TILING11"]
_T12_1_1, _T12_1_1_ROW, _T12_1_1_WIDTH = _SECTIONS["TILING12_1_1"]
_T12_1_1_, _T12_1_1__ROW, _T12_1_1__WIDTH = _SECTIONS["TILING12_1_1_"]
_T12_1_2, _T12_1_2_ROW, _T12_1_2_WIDTH = _SECTIONS["TILING12_1_2"]
_T12_2, _T12_2_ROW, _T12_2_WIDTH = _SECTIONS["TILING12_2"]
_T12_2_, _T12_2__ROW, _T12_2__WIDTH = _SECTIONS["TILING12_2_"]
_T13_1, _T13_1_ROW, _T13_1_WIDTH = _SECTIONS["TILING13_1"]
_T13_1_, _T13_1__ROW, _T13_1__WIDTH = _SECTIONS["TILING13_1_"]
_T13_2, _T13_2_ROW, _T13_2_WIDTH = _SECTIONS["TILING13_2"]
_T13_2_, _T13_2__ROW, _T13_2__WIDTH = _SECTIONS["TILING13_2_"]
_T13_3, _T13_3_ROW, _T13_3_WIDTH = _SECTIONS["TILING13_3"]
_T13_3_, _T13_3__ROW, _T13_3__WIDTH = _SECTIONS["TILING13_3_"]
_T13_4, _T13_4_ROW, _T13_4_WIDTH = _SECTIONS["TILING13_4"]
_T13_5_1, _T13_5_1_ROW, _T13_5_1_WIDTH = _SECTIONS["TILING13_5_1"]
_T13_5_2, _T13_5_2_ROW, _T13_5_2_WIDTH = _SECTIONS["TILING13_5_2"]
_T14, _T14_ROW, _T14_WIDTH = _SECTIONS["TILING14"]

vec8d = wp.types.vector(length=8, dtype=wp.float64)


@wp.func
def mc33_corner(c: wp.int32) -> wp.vec3i:
    # Corner ``c``'s offset from its cell's lowest node.
    y = (c >> 1) & 1
    return wp.vec3i((c & 1) ^ y, y, (c >> 2) & 1)


@wp.func
def tiling(
    base: wp.int32,
    row: wp.int32,
    width: wp.int32,
    config: wp.int32,
    sub: wp.int32,
    centre: wp.int32,
) -> wp.vec3i:
    # A tiling's first entry, entry count and whether it uses the cell vertex.
    return wp.vec3i(base + config * row + sub * width, width, centre)


@wp.func
def mc33_face_test(table: wp.array[wp.int32], values: vec8d, face: wp.int32) -> wp.bool:
    # The asymptotic decider on one face: true when the face's two positive corners join through
    # its interior, the sign of ``face`` flipping the answer.
    row = _FACE + (wp.abs(face) - 1) * 4
    a = values[table[row]]
    b = values[table[row + 1]]
    c = values[table[row + 2]]
    d = values[table[row + 3]]
    return wp.float64(face) * a * (a * c - b * d) >= wp.float64(0.0)


@wp.func
def mc33_interior_test(
    table: wp.array[wp.int32], values: vec8d, case: wp.int32, edge: wp.int32, s: wp.int32
) -> wp.bool:
    # Whether the cell's components join through its interior, on the section of the cell where
    # the bilinear field along the cell's height (cases 4 and 10) or across a reference edge's
    # crossing (cases 6, 7, 12 and 13) is most likely to connect them; ``s`` names the answer
    # that keeps them apart.
    zero = wp.float64(0.0)
    at = zero
    bt = zero
    ct = zero
    dt = zero
    if case == 4 or case == 10:
        a = (values[4] - values[0]) * (values[6] - values[2]) - (values[7] - values[3]) * (
            values[5] - values[1]
        )
        b = (
            values[2] * (values[4] - values[0])
            + values[0] * (values[6] - values[2])
            - values[1] * (values[7] - values[3])
            - values[3] * (values[5] - values[1])
        )
        t = -b / (wp.float64(2.0) * a)
        if t < zero or t > wp.float64(1.0):
            return s > 0
        at = values[0] + (values[4] - values[0]) * t
        bt = values[3] + (values[7] - values[3]) * t
        ct = values[2] + (values[6] - values[2]) * t
        dt = values[1] + (values[5] - values[1]) * t
    else:
        row = _INTERIOR + edge * 8
        p = values[table[row]]
        t = p / (p - values[table[row + 1]])
        b0 = values[table[row + 2]]
        c0 = values[table[row + 4]]
        d0 = values[table[row + 6]]
        bt = b0 + (values[table[row + 3]] - b0) * t
        ct = c0 + (values[table[row + 5]] - c0) * t
        dt = d0 + (values[table[row + 7]] - d0) * t
    test = wp.int32(0)
    if at >= zero:
        test += 1
    if bt >= zero:
        test += 2
    if ct >= zero:
        test += 4
    if dt >= zero:
        test += 8
    if test == 5:
        if at * ct < bt * dt:
            return s > 0
        return s < 0
    if test == 10:
        if at * ct >= bt * dt:
            return s > 0
        return s < 0
    if test == 7 or test == 11 or test >= 13:
        return s < 0
    return s > 0


@wp.func
def mc33_tiling(table: wp.array[wp.int32], values: vec8d, cube: wp.int32) -> wp.vec3i:
    # The tiling one cell takes: its first table entry, its entry count (three a triangle, zero
    # for an empty cell) and ``1`` when it uses the cell vertex. ``cube`` has bit ``c`` set where
    # corner ``c`` is above the level.
    case = table[_CASES + 2 * cube]
    config = table[_CASES + 2 * cube + 1]
    result = wp.vec3i(0, 0, 0)
    if case == 1:
        result = tiling(_T1, _T1_ROW, _T1_WIDTH, config, 0, 0)
    elif case == 2:
        result = tiling(_T2, _T2_ROW, _T2_WIDTH, config, 0, 0)
    elif case == 3:
        if mc33_face_test(table, values, table[_TEST3 + config]):
            result = tiling(_T3_2, _T3_2_ROW, _T3_2_WIDTH, config, 0, 0)
        else:
            result = tiling(_T3_1, _T3_1_ROW, _T3_1_WIDTH, config, 0, 0)
    elif case == 4:
        if mc33_interior_test(table, values, case, 0, table[_TEST4 + config]):
            result = tiling(_T4_1, _T4_1_ROW, _T4_1_WIDTH, config, 0, 0)
        else:
            result = tiling(_T4_2, _T4_2_ROW, _T4_2_WIDTH, config, 0, 0)
    elif case == 5:
        result = tiling(_T5, _T5_ROW, _T5_WIDTH, config, 0, 0)
    elif case == 6:
        tests = _TEST6 + config * 3
        if mc33_face_test(table, values, table[tests]):
            result = tiling(_T6_2, _T6_2_ROW, _T6_2_WIDTH, config, 0, 0)
        elif mc33_interior_test(table, values, case, table[tests + 2], table[tests + 1]):
            result = tiling(_T6_1_1, _T6_1_1_ROW, _T6_1_1_WIDTH, config, 0, 0)
        else:
            result = tiling(_T6_1_2, _T6_1_2_ROW, _T6_1_2_WIDTH, config, 0, 1)
    elif case == 7:
        tests = _TEST7 + config * 5
        sub = wp.int32(0)
        for bit in range(3):
            if mc33_face_test(table, values, table[tests + bit]):
                sub += 1 << bit
        if sub == 0:
            result = tiling(_T7_1, _T7_1_ROW, _T7_1_WIDTH, config, 0, 0)
        elif sub == 1 or sub == 2:
            result = tiling(_T7_2, _T7_2_ROW, _T7_2_WIDTH, config, sub - 1, 0)
        elif sub == 4:
            result = tiling(_T7_2, _T7_2_ROW, _T7_2_WIDTH, config, 2, 0)
        elif sub == 3:
            result = tiling(_T7_3, _T7_3_ROW, _T7_3_WIDTH, config, 0, 1)
        elif sub == 5 or sub == 6:
            result = tiling(_T7_3, _T7_3_ROW, _T7_3_WIDTH, config, sub - 4, 1)
        elif mc33_interior_test(table, values, case, table[tests + 4], table[tests + 3]):
            result = tiling(_T7_4_2, _T7_4_2_ROW, _T7_4_2_WIDTH, config, 0, 0)
        else:
            result = tiling(_T7_4_1, _T7_4_1_ROW, _T7_4_1_WIDTH, config, 0, 0)
    elif case == 8:
        result = tiling(_T8, _T8_ROW, _T8_WIDTH, config, 0, 0)
    elif case == 9:
        result = tiling(_T9, _T9_ROW, _T9_WIDTH, config, 0, 0)
    elif case == 10:
        tests = _TEST10 + config * 3
        if mc33_face_test(table, values, table[tests]):
            if mc33_face_test(table, values, table[tests + 1]):
                result = tiling(_T10_1_1_, _T10_1_1__ROW, _T10_1_1__WIDTH, config, 0, 0)
            else:
                result = tiling(_T10_2, _T10_2_ROW, _T10_2_WIDTH, config, 0, 1)
        elif mc33_face_test(table, values, table[tests + 1]):
            result = tiling(_T10_2_, _T10_2__ROW, _T10_2__WIDTH, config, 0, 1)
        elif mc33_interior_test(table, values, case, 0, table[tests + 2]):
            result = tiling(_T10_1_1, _T10_1_1_ROW, _T10_1_1_WIDTH, config, 0, 0)
        else:
            result = tiling(_T10_1_2, _T10_1_2_ROW, _T10_1_2_WIDTH, config, 0, 0)
    elif case == 11:
        result = tiling(_T11, _T11_ROW, _T11_WIDTH, config, 0, 0)
    elif case == 12:
        tests = _TEST12 + config * 4
        if mc33_face_test(table, values, table[tests]):
            if mc33_face_test(table, values, table[tests + 1]):
                result = tiling(_T12_1_1_, _T12_1_1__ROW, _T12_1_1__WIDTH, config, 0, 0)
            else:
                result = tiling(_T12_2, _T12_2_ROW, _T12_2_WIDTH, config, 0, 1)
        elif mc33_face_test(table, values, table[tests + 1]):
            result = tiling(_T12_2_, _T12_2__ROW, _T12_2__WIDTH, config, 0, 1)
        elif mc33_interior_test(table, values, case, table[tests + 3], table[tests + 2]):
            result = tiling(_T12_1_1, _T12_1_1_ROW, _T12_1_1_WIDTH, config, 0, 0)
        else:
            result = tiling(_T12_1_2, _T12_1_2_ROW, _T12_1_2_WIDTH, config, 0, 0)
    elif case == 13:
        tests = _TEST13 + config * 7
        sub = wp.int32(0)
        for bit in range(6):
            if mc33_face_test(table, values, table[tests + bit]):
                sub += 1 << bit
        sub = table[_SUBCONFIG13 + sub]
        if sub == 0:
            result = tiling(_T13_1, _T13_1_ROW, _T13_1_WIDTH, config, 0, 0)
        elif sub <= 6:
            result = tiling(_T13_2, _T13_2_ROW, _T13_2_WIDTH, config, sub - 1, 0)
        elif sub <= 18:
            result = tiling(_T13_3, _T13_3_ROW, _T13_3_WIDTH, config, sub - 7, 1)
        elif sub <= 22:
            result = tiling(_T13_4, _T13_4_ROW, _T13_4_WIDTH, config, sub - 19, 1)
        elif sub <= 26:
            first = tiling(_T13_5_1, _T13_5_1_ROW, _T13_5_1_WIDTH, config, sub - 23, 0)
            if mc33_interior_test(table, values, case, table[first[0]], table[tests + 6]):
                result = first
            else:
                result = tiling(_T13_5_2, _T13_5_2_ROW, _T13_5_2_WIDTH, config, sub - 23, 0)
        elif sub <= 38:
            result = tiling(_T13_3_, _T13_3__ROW, _T13_3__WIDTH, config, sub - 27, 1)
        elif sub <= 44:
            result = tiling(_T13_2_, _T13_2__ROW, _T13_2__WIDTH, config, sub - 39, 0)
        else:
            result = tiling(_T13_1_, _T13_1__ROW, _T13_1__WIDTH, config, 0, 0)
    elif case == 14:
        result = tiling(_T14, _T14_ROW, _T14_WIDTH, config, 0, 0)
    return result


@wp.func
def mc33_cell_good(field: wp.array3d[wp.float64], cell: wp.vec3i) -> wp.bool:
    # Whether ``cell`` is inside the lattice and all eight of its corners carry a value: only such
    # a cell is triangulated, so a NaN node leaves a hole around it.
    for axis in range(3):
        if cell[axis] < 0 or cell[axis] >= field.shape[axis] - 1:
            return False
    for c in range(8):
        node = cell + mc33_corner(c)
        if wp.isnan(field[node[0], node[1], node[2]]):
            return False
    return True


@wp.func
def mc33_edge_crosses(field: wp.array3d[wp.float64], node: wp.vec3i, axis: wp.int32) -> wp.bool:
    # Whether the edge from ``node`` along ``axis`` joins a node above the level (``> 0``) to one
    # that is not. Its far end must be in the lattice; a NaN end never crosses.
    far = node
    far[axis] = far[axis] + 1
    if far[axis] >= field.shape[axis]:
        return False
    a = field[node[0], node[1], node[2]]
    b = field[far[0], far[1], far[2]]
    if wp.isnan(a) or wp.isnan(b):
        return False
    return (a > wp.float64(0.0)) != (b > wp.float64(0.0))


@wp.func
def mc33_sharing_cell(node: wp.vec3i, axis: wp.int32, slot: wp.int32) -> wp.vec3i:
    # The ``slot``-th (0-3) of the four cells sharing the edge from ``node`` along ``axis``.
    cell = node
    u = (axis + 1) % 3
    w = (axis + 2) % 3
    cell[u] = cell[u] - (slot & 1)
    cell[w] = cell[w] - ((slot >> 1) & 1)
    return cell


@wp.func
def mc33_edge_has_vertex(field: wp.array3d[wp.float64], node: wp.vec3i, axis: wp.int32) -> wp.bool:
    # A crossing edge carries a vertex once a triangulated cell shares it.
    if not mc33_edge_crosses(field, node, axis):
        return False
    for slot in range(4):
        if mc33_cell_good(field, mc33_sharing_cell(node, axis, slot)):
            return True
    return False


@wp.func
def mc33_walked_before(a: wp.vec3i, b: wp.vec3i) -> wp.bool:
    # Whether a serial walk reaches cell ``a`` before cell ``b``: slices along ``y``, then rows
    # along ``x``, then ``z``.
    if a[1] != b[1]:
        return a[1] < b[1]
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[2] < b[2]


@wp.func
def mc33_edge_vertex(field: wp.array3d[wp.float64], node: wp.vec3i, axis: wp.int32) -> wp.vec3d:
    # The crossing on the edge from ``node`` along ``axis``, in lattice coordinates: linear in the
    # field, from the edge's lower end.
    far = node
    far[axis] = far[axis] + 1
    f1 = field[node[0], node[1], node[2]]
    u = f1 / (f1 - field[far[0], far[1], far[2]])
    p = wp.vec3d(wp.float64(node[0]), wp.float64(node[1]), wp.float64(node[2]))
    p[axis] = p[axis] * (wp.float64(1.0) - u) + u * wp.float64(far[axis])
    return p


@wp.func
def mc33_cell_vertex(
    table: wp.array[wp.int32], field: wp.array3d[wp.float64], cell: wp.vec3i
) -> wp.vec3d:
    # The vertex a subcase adds inside ``cell``, in lattice coordinates: the mean of the crossings
    # on the cell's edges, in edge order, that a serial walk (``mc33_walked_before``) has made
    # before it reaches the cell -- those a triangulated cell walked earlier shares. A crossing
    # only this cell or a later one makes is not yet there and is left out, and a cell none of
    # whose crossings is made yet takes the ``0 / 0`` mean, NaN. Read off the walk order, not
    # kept from a walk, so it does not depend on the order the cells run in.
    total = wp.vec3d()
    count = wp.int32(0)
    for e in range(12):
        row = _EDGES + 2 * e
        low = mc33_corner(table[row])
        high = mc33_corner(table[row + 1])
        axis = wp.int32(0)
        if low[1] != high[1]:
            axis = 1
        elif low[2] != high[2]:
            axis = 2
        node = cell + low
        made = False
        if mc33_edge_crosses(field, node, axis):
            for slot in range(4):
                other = mc33_sharing_cell(node, axis, slot)
                if mc33_walked_before(other, cell) and mc33_cell_good(field, other):
                    made = True
        if made:
            total += mc33_edge_vertex(field, node, axis)
            count += 1
    return total / wp.float64(count)


@wp.func
def mc33_cell_tiling(
    table: wp.array[wp.int32], field: wp.array3d[wp.float64], cell: wp.vec3i
) -> wp.vec3i:
    # ``mc33_tiling`` of ``cell``, empty for a cell outside the lattice or with a NaN corner.
    if not mc33_cell_good(field, cell):
        return wp.vec3i(0, 0, 0)
    values = vec8d()
    cube = wp.int32(0)
    for c in range(8):
        node = cell + mc33_corner(c)
        value = field[node[0], node[1], node[2]]
        values[c] = value
        if value > wp.float64(0.0):
            cube += 1 << c
    return mc33_tiling(table, values, cube)


@wp.func
def mc33_vertex_mask(field: wp.array3d[wp.float64], node: wp.vec3i) -> wp.int32:
    # Bit ``axis`` set where the edge from ``node`` along ``axis`` carries a vertex.
    mask = wp.int32(0)
    for axis in range(3):
        if mc33_edge_has_vertex(field, node, axis):
            mask |= 1 << axis
    return mask


@wp.func
def mc33_bits_from(mask: wp.int32, axis: wp.int32) -> wp.int32:
    # How many of ``mask``'s three bits are at ``axis`` or above.
    rest = mask >> axis
    return (rest & 1) + ((rest >> 1) & 1) + ((rest >> 2) & 1)


@wp.kernel
def mc33_counts(
    field: wp.array3d[wp.float64],
    table: wp.array[wp.int32],
    out_counts: wp.array3d[wp.vec3i],
    out_masks: wp.array3d[wp.uint8],
) -> None:
    """
    Each node's edge vertices, triangles and cell vertex, and which of its edges carry a vertex.

    A node owns its three positive-axis edges and the cell it is the lowest corner of. The counts
    are scanned inclusively in place, in node order, to number the vertices and triangles.
    """
    i, j, k = wp.tid()
    node = wp.vec3i(i, j, k)
    mask = mc33_vertex_mask(field, node)
    tiles = mc33_cell_tiling(table, field, node)
    out_counts[i, j, k] = wp.vec3i(mc33_bits_from(mask, 0), tiles[1] // 3, tiles[2])
    out_masks[i, j, k] = wp.uint8(mask)


@wp.kernel
def mc33_emit(
    field: wp.array3d[wp.float64],
    table: wp.array[wp.int32],
    counts: wp.array3d[wp.vec3i],
    masks: wp.array3d[wp.uint8],
    lower: wp.vec3d,
    spacing: wp.vec3d,
    n_edge_vertices: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    """
    Write each node's edge vertices, and its cell's triangles and cell vertex.

    ``counts`` is ``mc33_counts``' output scanned inclusively. Edge vertices come first, by node
    and then by axis; cell vertices follow them, by node. A lattice point ``p`` is placed at
    ``lower + p * spacing``.
    """
    i, j, k = wp.tid()
    node = wp.vec3i(i, j, k)
    mask = wp.int32(masks[i, j, k])
    scanned = counts[i, j, k]
    slot = scanned[0] - mc33_bits_from(mask, 0)
    for axis in range(3):
        if (mask >> axis) & 1 != 0:
            p = mc33_edge_vertex(field, node, axis)
            out_vertices[slot] = wp.vec3(wp.cw_mul(p, spacing) + lower)
            slot += 1
    tiles = mc33_cell_tiling(table, field, node)
    if tiles[1] == 0:
        return
    centre = n_edge_vertices + scanned[2] - tiles[2]
    if tiles[2] != 0:
        p = mc33_cell_vertex(table, field, node)
        out_vertices[centre] = wp.vec3(wp.cw_mul(p, spacing) + lower)
    corner = 3 * (scanned[1] - tiles[1] // 3)
    for entry in range(tiles[1]):
        e = table[tiles[0] + entry]
        vertex = centre
        if e < 12:
            row = _EDGES + 2 * e
            low = mc33_corner(table[row])
            high = mc33_corner(table[row + 1])
            axis = wp.int32(0)
            if low[1] != high[1]:
                axis = 1
            elif low[2] != high[2]:
                axis = 2
            owner = node + low
            vertex = counts[owner[0], owner[1], owner[2]][0] - mc33_bits_from(
                wp.int32(masks[owner[0], owner[1], owner[2]]), axis
            )
        out_faces[corner + entry] = vertex
