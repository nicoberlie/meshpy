import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la


def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]


def main():
    
    ## Domain dimensions and determinations facets for markers (left, right, down, up)
    points = [(1, 0.7), (0, 0.7), (0, 0), (1, 0)]
    #facets = round_trip_connect(0, len(points) - 1)
    facets = [(0, 1), (1, 2), (0, 3), (2, 3)]
    
    # circ_start = len(points)
    # points.extend(
    #     (3 * np.cos(angle), 3 * np.sin(angle))
    #     for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False)
    # )

    # facets.extend(round_trip_connect(circ_start, len(points) - 1))
    
    ##Grid refinement
    
    def needs_refinement(vertices, area):
        #bary = np.sum(np.array(vertices), axis=0) / 3
        max_area = 1e-4# + (la.norm(bary, np.inf) - 1) * 0.01
        return bool(area > max_area)
    
    ## Grid information structure creation

    info = triangle.MeshInfo()
    info.set_points(points)
    #info.set_holes([(2, 0)])
    info.set_facets(facets,[1,2,3,4])
    
    ## Mesh build

    mesh = triangle.build(info, refinement_func=needs_refinement,generate_faces=True,attributes=True,mesh_order=2)
    #mesh = triangle.build(info,generate_faces=True,attributes=True,mesh_order=2)
    
    # Define order
    
    order = np.array(3)
    
    ##Extract mesh info

    nodes0   = np.array(mesh.points)
    elems0   = np.array(mesh.elements)
    faces0  = np.array(mesh.faces)
    marks = np.array(mesh.face_markers)
    #sattr   = np.array(mesh.element_attributes)
    
    ## Add midpoints to faces
    
    faces = np.zeros([len(faces0), 3], dtype=np.int64)
    faces[:,:2] = faces0
    for i in range(0,len(faces)):
        idn = np.max(faces0)+1+i
        faces[i,2] = idn
        
    ## Keep only boundary faces and markers

    #faces = faces0
    faces = np.delete(faces,np.where(marks==0),0)
    marks = np.delete(marks,np.where(marks==0),0)
        
        
    ## Add barycenters nodes to "nodes" and "elems"
    nodes = np.zeros([len(nodes0)+len(elems0),2])
    nodes[0:len(nodes0),:] = nodes0
    elems = np.zeros([len(elems0),7], dtype=np.int64)
    elems[:,:6] = elems0
    
    for i in range(0,len(elems0)):
        idn  = elems0[i,:3]
        newx = np.mean(nodes0[idn,0])
        newy = np.mean(nodes0[idn,1])
        nodes[len(nodes0)+i,:] = [newx,newy]
        elems[i,6] = len(nodes0)+i
        
    ## Moving the mid-center points to fit plast code (shape functions)
    
    elems[:,[3,4,5]] = elems[:,[5,3,4]]
        
    ## Get number corner nodes
    
    cn = elems[:,:3]
    cn = cn.flatten()
    cn = np.unique(cn)
    nc = np.array(len(cn))
    
    ## Get attribute (one attribute for each element, phase defined by position of center node)
    # 1 is matrix, 2 is elastic anomaly
    
    att = np.ones(len(elems), dtype=np.int32)
    nodec = nodes[elems[:,6]]
    rad = nodec[:,0]**2 + nodec[:,1]**2
    att[rad < 0.05**2] = 2
        
    
    ## Testing plots
        

    import matplotlib.pyplot as plt
    
    # i = 120
    # idf = faces[i]
    # plt.plot(nodes0[idf, 0], nodes0[idf, 1], "ro")
    
    #fb = faces[np.where(marks==1),:]
    #fb = fb.flatten()
    #fb = np.unique(fb)
    #plt.plot(nodes[fb, 0], nodes[fb, 1], "ro")
    
    #plt.plot(nodes[cn, 0], nodes[cn, 1], "ro")
    
    plt.plot(nodes[elems[att==2,6],0],nodes[elems[att==2,6],1], "ro")

    idcn = np.max(faces0)+1
    plt.triplot(nodes0[0:idcn, 0], nodes0[0:idcn, 1], elems0[:,0:3])
    plt.show()
    
    ## save values in npz format
    
    np.savez_compressed('BM_mesh', nc=nc, order=order, nodes=nodes, elems=elems, faces=faces, marks=marks, att=att)


if __name__ == "__main__":
    main()
