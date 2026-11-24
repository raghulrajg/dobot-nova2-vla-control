# Products

One folder per product. The folder name is the name you use in a task:

```
objects/
  gear_housing/
    ref_01.png  ref_02.png  ref_03.png ...   # reference photos (.jpg .png .bmp .webp)
    object.json                               # optional hints
```

    pick the object 'gear_housing' place the 'bin_A'

A new product needs only a new folder: nothing is retrained or reprogrammed.
The finder compares every object it sees on the table with these photos
(DINOv2 features) and picks the best match above `min_similarity`.

## Reference photos

Best: take them in the cell, with the wrist camera at the detection pose,
so lighting, scale and viewpoint match what the robot sees:

```bash
python -m nova2_vla capture-refs gear_housing --views 4
```

It asks you to put the product alone under the camera, turn it between
shots, and saves a crop of it each time. 3–6 views is plenty; add views of
the sides it can lie on. Photos from elsewhere work too, cropped tightly to
the product.

If two products look alike, check `runs/<task>/attempt1.jpg` (every
candidate's similarity is drawn on it) and add photos that show the
difference, or raise `min_similarity` for that product.

## object.json (all optional)

| Key | Meaning | Default |
|---|---|---|
| `description` | words for the VLA's instruction (`pick up the {description}`) | folder name |
| `grasp_depth_m` | fingertips this far below the top (capped at half the height) | `pick.grasp_depth_m` |
| `grasp_yaw_offset_deg` | turn the scripted grasp about the vertical | `pick.grasp_yaw_offset_deg` |
| `grip_torque` | Feetech torque limit 0..1000 for this product | `gripper.torque_limit` |
| `min_similarity` | match threshold for this product | `perception.min_similarity` |

`example_part/` only shows the format: it has no photos, so it is not
listed until you add some.
