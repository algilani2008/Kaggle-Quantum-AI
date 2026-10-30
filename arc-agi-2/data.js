window.ARC_DATA = {
 "training": {
  "tasks": 1000,
  "programs": 105,
  "fit": 82,
  "accuracy": 6.9,
  "solved": 69.0,
  "runtime": 34.3,
  "byFamily": [
   [
    "grid:combine_cells",
    14
   ],
   [
    "pattern:tile2x2",
    9
   ],
   [
    "geom:identity+colormap",
    4
   ],
   [
    "pattern:periodic_repair",
    3
   ],
   [
    "pattern:tile2x1",
    3
   ],
   [
    "shape:recolor_by_size",
    3
   ],
   [
    "pattern:tile3x3",
    2
   ],
   [
    "shape:recolor_by_size_rank",
    2
   ],
   [
    "shape:select_largest",
    2
   ],
   [
    "grid:select_odd",
    2
   ],
   [
    "geom:rot180",
    2
   ],
   [
    "scale_up:(2, 2)",
    2
   ],
   [
    "pattern:tile1x2",
    2
   ],
   [
    "geom:transpose",
    2
   ],
   [
    "crop_nonbg",
    1
   ],
   [
    "shape:recolor_by_size_8c",
    1
   ],
   [
    "shape:select_smallest",
    1
   ],
   [
    "scale_down:(3, 3)+colormap",
    1
   ],
   [
    "pattern:tile1x4",
    1
   ],
   [
    "geom:flip_lr",
    1
   ],
   [
    "geom:flip_ud",
    1
   ],
   [
    "scale_down:(2, 2)",
    1
   ],
   [
    "crop_nonbg>flip_lr",
    1
   ],
   [
    "scale_up:(3, 3)",
    1
   ],
   [
    "crop_nonbg+colormap",
    1
   ],
   [
    "shape:recolor_by_shape",
    1
   ],
   [
    "shape:recolor_by_size_rank_8c",
    1
   ],
   [
    "pattern:symmetry_repair",
    1
   ],
   [
    "pattern:tile1x5",
    1
   ],
   [
    "shape:select_unique_color_mc",
    1
   ],
   [
    "pattern:tile4x4",
    1
   ],
   [
    "geom:rot90",
    1
   ]
  ],
  "fits": [
   [
    "shape:recolor_by_shape",
    16
   ],
   [
    "shape:recolor_by_shape_8c",
    14
   ],
   [
    "grid:combine_cells",
    14
   ],
   [
    "shape:recolor_by_size",
    9
   ],
   [
    "pattern:tile2x2",
    9
   ],
   [
    "pattern:tile2x2>rot180",
    9
   ],
   [
    "shape:recolor_by_size_8c",
    8
   ],
   [
    "pattern:tile2x2>flip_ud",
    8
   ],
   [
    "pattern:tile2x2>flip_lr",
    8
   ],
   [
    "shape:recolor_by_size_rank",
    7
   ],
   [
    "pattern:tile2x2>rot90",
    7
   ],
   [
    "pattern:tile2x2>rot270",
    7
   ],
   [
    "shape:recolor_by_size_rank_desc",
    6
   ],
   [
    "shape:recolor_by_box_8c",
    6
   ],
   [
    "shape:recolor_by_box",
    6
   ]
  ]
 },
 "evaluation": {
  "tasks": 120,
  "programs": 105,
  "fit": 0,
  "accuracy": 0.0,
  "solved": 0.0,
  "runtime": 7.1,
  "byFamily": [],
  "fits": []
 },
 "gemini": {
  "model": "gemini-4-argon",
  "attempted": 36,
  "verified": 32,
  "testCorrect": 32.0,
  "tokensIn": "247,401",
  "tokensOut": "775,747",
  "tasks": [
   {
    "id": "0934a4d8",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T13:45:17.016198+00:00",
    "rounds": 2,
    "files": [
     "0934a4d8"
    ],
    "lines": 73
   },
   {
    "id": "135a2760",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T13:49:51.784159+00:00",
    "rounds": 2,
    "files": [
     "135a2760"
    ],
    "lines": 73
   },
   {
    "id": "136b0064",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T13:54:06.322287+00:00",
    "rounds": 2,
    "files": [
     "136b0064"
    ],
    "lines": 37
   },
   {
    "id": "13e47133",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T13:57:33.920914+00:00",
    "rounds": 2,
    "files": [
     "13e47133"
    ],
    "lines": 64
   },
   {
    "id": "142ca369",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:05:37.978900+00:00",
    "rounds": 6,
    "files": [
     "142ca369"
    ],
    "lines": 70
   },
   {
    "id": "16b78196",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:15:10.020617+00:00",
    "rounds": 3,
    "files": [
     "16b78196"
    ],
    "lines": 157
   },
   {
    "id": "16de56c4",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:17:29.418588+00:00",
    "rounds": 2,
    "files": [
     "16de56c4"
    ],
    "lines": 50
   },
   {
    "id": "1818057f",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:19:52.792883+00:00",
    "rounds": 2,
    "files": [
     "1818057f"
    ],
    "lines": 26
   },
   {
    "id": "195c6913",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:23:58.958628+00:00",
    "rounds": 2,
    "files": [
     "195c6913"
    ],
    "lines": 69
   },
   {
    "id": "1ae2feb7",
    "status": "failed",
    "test": 0.0,
    "at": "2026-10-23T14:28:11.452978+00:00",
    "rounds": 6,
    "files": [],
    "lines": 0
   },
   {
    "id": "20270e3b",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:34:12.643445+00:00",
    "rounds": 2,
    "files": [
     "20270e3b"
    ],
    "lines": 94
   },
   {
    "id": "20a9e565",
    "status": "failed",
    "test": 0.0,
    "at": "2026-10-23T14:38:15.437366+00:00",
    "rounds": 6,
    "files": [],
    "lines": 0
   },
   {
    "id": "21897d95",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:47:33.029647+00:00",
    "rounds": 3,
    "files": [
     "21897d95"
    ],
    "lines": 136
   },
   {
    "id": "221dfab4",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:50:43.915642+00:00",
    "rounds": 2,
    "files": [
     "221dfab4"
    ],
    "lines": 32
   },
   {
    "id": "247ef758",
    "status": "failed",
    "test": 0.0,
    "at": "2026-10-23T14:54:53.895226+00:00",
    "rounds": 6,
    "files": [],
    "lines": 0
   },
   {
    "id": "269e22fb",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T14:59:26.096548+00:00",
    "rounds": 2,
    "files": [
     "269e22fb"
    ],
    "lines": 50
   },
   {
    "id": "271d71e2",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T15:58:33.094172+00:00",
    "rounds": 2,
    "files": [
     "271d71e2"
    ],
    "lines": 109
   },
   {
    "id": "28a6681f",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T16:06:08.336253+00:00",
    "rounds": 2,
    "files": [
     "28a6681f"
    ],
    "lines": 63
   },
   {
    "id": "291dc1e1",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T16:11:43.360867+00:00",
    "rounds": 2,
    "files": [
     "291dc1e1"
    ],
    "lines": 111
   },
   {
    "id": "2b83f449",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T16:31:47.336454+00:00",
    "rounds": 4,
    "files": [
     "2b83f449"
    ],
    "lines": 55
   },
   {
    "id": "2ba387bc",
    "status": "failed",
    "test": 0.0,
    "at": "2026-10-23T16:36:07.761333+00:00",
    "rounds": 6,
    "files": [],
    "lines": 0
   },
   {
    "id": "2c181942",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T16:41:10.743522+00:00",
    "rounds": 2,
    "files": [
     "2c181942"
    ],
    "lines": 85
   },
   {
    "id": "2d0172a1",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T16:50:03.599915+00:00",
    "rounds": 3,
    "files": [
     "2d0172a1"
    ],
    "lines": 151
   },
   {
    "id": "31f7f899",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T16:52:00.499154+00:00",
    "rounds": 2,
    "files": [
     "31f7f899",
     "31f7f899_2"
    ],
    "lines": 99
   },
   {
    "id": "332f06d7",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T16:55:03.246872+00:00",
    "rounds": 2,
    "files": [
     "332f06d7"
    ],
    "lines": 69
   },
   {
    "id": "35ab12c3",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:01:01.705158+00:00",
    "rounds": 4,
    "files": [
     "35ab12c3"
    ],
    "lines": 146
   },
   {
    "id": "36a08778",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:04:28.333164+00:00",
    "rounds": 2,
    "files": [
     "36a08778"
    ],
    "lines": 33
   },
   {
    "id": "38007db0",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:05:43.862916+00:00",
    "rounds": 2,
    "files": [
     "38007db0"
    ],
    "lines": 34
   },
   {
    "id": "3a25b0d8",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:15:06.399155+00:00",
    "rounds": 2,
    "files": [
     "3a25b0d8"
    ],
    "lines": 127
   },
   {
    "id": "3dc255db",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:29:13.668045+00:00",
    "rounds": 3,
    "files": [
     "3dc255db"
    ],
    "lines": 108
   },
   {
    "id": "3e6067c3",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:33:15.503126+00:00",
    "rounds": 2,
    "files": [
     "3e6067c3"
    ],
    "lines": 123
   },
   {
    "id": "409aa875",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:37:16.491438+00:00",
    "rounds": 2,
    "files": [
     "409aa875"
    ],
    "lines": 63
   },
   {
    "id": "446ef5d2",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:48:38.341845+00:00",
    "rounds": 3,
    "files": [
     "446ef5d2"
    ],
    "lines": 177
   },
   {
    "id": "45a5af55",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:49:55.898650+00:00",
    "rounds": 2,
    "files": [
     "45a5af55"
    ],
    "lines": 10
   },
   {
    "id": "4a21e3da",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:55:45.092403+00:00",
    "rounds": 2,
    "files": [
     "4a21e3da"
    ],
    "lines": 91
   },
   {
    "id": "4c3d4a41",
    "status": "verified",
    "test": 1.0,
    "at": "2026-10-23T17:59:37.132154+00:00",
    "rounds": 2,
    "files": [
     "4c3d4a41"
    ],
    "lines": 49
   }
  ]
 }
};
