# ARC-AGI-2 solver report — 2026-10-23T12:55:55.027660+00:00

- Source: `/tmp/claude-0/-home-claude/bad02fd1-1ea7-58a3-aae9-5e26fef0f1e9/scratchpad/arc2/data/training`
- Tasks: 1000; DSL programs in library: 105
- Tasks where at least one program fit all train pairs: 82 (8.2%)
- **pass@2 accuracy: 6.90%** (69.00 / 1000 tasks)

## Correct predictions by DSL family

- grid:combine_cells: 14
- pattern:tile2x2: 9
- geom:identity+colormap: 4
- pattern:periodic_repair: 3
- pattern:tile2x1: 3
- shape:recolor_by_size: 3
- pattern:tile3x3: 2
- shape:recolor_by_size_rank: 2
- shape:select_largest: 2
- grid:select_odd: 2
- geom:rot180: 2
- scale_up:(2, 2): 2
- pattern:tile1x2: 2
- geom:transpose: 2
- crop_nonbg: 1
- shape:recolor_by_size_8c: 1
- shape:select_smallest: 1
- scale_down:(3, 3)+colormap: 1
- pattern:tile1x4: 1
- geom:flip_lr: 1
- geom:flip_ud: 1
- scale_down:(2, 2): 1
- crop_nonbg>flip_lr: 1
- scale_up:(3, 3): 1
- crop_nonbg+colormap: 1
- shape:recolor_by_shape: 1
- shape:recolor_by_size_rank_8c: 1
- pattern:symmetry_repair: 1
- pattern:tile1x5: 1
- shape:select_unique_color_mc: 1
- pattern:tile4x4: 1
- geom:rot90: 1

## Programs that fit train pairs (top 15)

- shape:recolor_by_shape: 16
- shape:recolor_by_shape_8c: 14
- grid:combine_cells: 14
- shape:recolor_by_size: 9
- pattern:tile2x2: 9
- pattern:tile2x2>rot180: 9
- shape:recolor_by_size_8c: 8
- pattern:tile2x2>flip_ud: 8
- pattern:tile2x2>flip_lr: 8
- shape:recolor_by_size_rank: 7
- pattern:tile2x2>rot90: 7
- pattern:tile2x2>rot270: 7
- shape:recolor_by_size_rank_desc: 6
- shape:recolor_by_box_8c: 6
- shape:recolor_by_box: 6

Runtime: 34.3s
