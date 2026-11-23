# Sample CAD files

Both files follow the DetectorPlace layer convention (see `detectorplace/cad.py`):

| Layer | Content |
|---|---|
| `EQUIP*` | Closed polylines or circles. Height = entity thickness, or a layer suffix such as `EQUIP_H9`. |
| `HAZ_A`, `HAZ_B`, `HAZ_C` | Closed polylines: fire hazard Grade A / B / C areas |
| `LEAK*` | Points, circles or block inserts: credible gas leak sources |
| `FG_PANEL` | Point: F&G panel (cable routing origin) |
| `PLOT_LIMIT` | Optional closed polyline: the analysis boundary |
| anything else | Ignored and listed in the read report (grid, text, dimensions, title block) |

* `demo_plant.dxf`: the demo gas compression station, drawn in millimetres. Reads back to the same
  model as the original (hazard grades 100%, heights 99.9% of cells; circles rasterise slightly differently).
* `second_plant.dxf`: a different invented plant in metres, with deliberate problems for the quality checks:
  one outline has no height, one takes its height from the layer name (`EQUIP_H9`), and there is no `FG_PANEL`.

All geometry is synthetic.
