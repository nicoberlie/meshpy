# Quadratic element demo, by Aravind Alwan

import numpy as np
import matplotlib.pyplot as plt
from meshpy.triangle import MeshInfo, build

# Utility function to create lists of the form [(1,2), (2,3), (3,4),
# (4,1)], given two numbers 1 and 4
from itertools import islice, cycle


def loop(a, b):
    return list(
            zip(list(range(a, b)), islice(cycle(list(range(a, b))), 1, None))
            )


info = MeshInfo()
info.set_points([(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (3, 0), (3, 1), (2, 1)])
info.set_facets(
    loop(0, 4) + loop(4, 8), list(range(1, 9))
)  # Create 8 facets and apply markers 1-8 on them
info.regions.resize(2)
info.regions[0] = [
    0.5,
    0.5,
    1,
    0.1,
]  # Fourth item specifies maximum area of triangles as a region attribute
info.regions[1] = [
    2.5,
    0.5,
    2,
    0.1,
]  # Replace 0.1 by a smaller value to produce a finer mesh

mesh = build(
    info,
    volume_constraints=True,
    attributes=True,
    generate_faces=True,
    min_angle=33,
    mesh_order=2,
)

pts = np.vstack(mesh.points)  # (npoints, 2)-array of points
elements = np.vstack(
    mesh.elements
)  # (ntriangles, 6)-array specifying element connectivity

# Matplotlib's Triangulation module uses only linear elements, so use only
# first 3 columns when plotting
#plt.triplot(pts[:, 0], pts[:, 1], elements[:, :3])

plt.plot(
    pts[:, 0], pts[:, 1], "ko"
)  # Manually plot all points including the ones at the midpoints of triangle faces

faces2 = np.vstack(mesh.faces)
faces = np.zeros([len(faces2), 3], dtype=np.int64)
faces[:,:2] = faces2
for i in range(0,len(faces)):
    idn = np.max(faces2)+1+i
    faces[i,2] = idn
    
markers = np.array(mesh.face_markers)
    
# i = 50
# idf = faces[i]
# plt.plot(
#      pts[idf, 0], pts[idf, 1], "ro"
#     )

fb = faces[np.where(markers==8),:]
fb = fb.flatten()
fb = np.unique(fb)
plt.plot(pts[fb, 0], pts[fb, 1], "ro")

# Plot a filled contour plot of the function (x - 1.5)^2 + y^2 over
# the mesh. Note tricontourf interpolation uses only linear elements
# plt.tricontourf(
#     pts[:, 0], pts[:, 1], elements[:, :3],
#     (pts[:, 0] - 1.5) ** 2 + pts[:, 1] ** 2,
#     100)



plt.triplot(pts[:, 0], pts[:, 1], elements[:,:3])

#plt.show()

plt.axis([-0.1, 3.1, -0.8, 1.8])
plt.show()
