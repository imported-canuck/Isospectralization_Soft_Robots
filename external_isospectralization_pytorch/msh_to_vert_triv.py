#!/usr/bin/env python3
import sys
import os
import gmsh

def main():
    if len(sys.argv) < 2:
        print("Usage: python msh_to_vert_triv.py input.msh [output_base]")
        sys.exit(1)

    in_msh = sys.argv[1]
    out_base = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(in_msh)[0]

    gmsh.initialize()
    try:
        gmsh.open(in_msh)

        # Gather all surface entities (dimension = 2)
        surf_ents = gmsh.model.getEntities(2)  # list of (dim=2, tag)
        triangles = []        # list of triples of original node tags
        used_nodes = set()    # set of node tags that appear in triangles

        # Helper: collect triangle connectivity from a single surface
        def collect_from_surface(sTag):
            # getElements returns lists aligned per element type in this entity
            etypes, etags, enodes = gmsh.model.mesh.getElements(2, sTag)
            for etype, nodes_flat in zip(etypes, enodes):
                # Linear 3-node triangle is type 2; high-order triangles are 9 (6-node), 21 (10-node), etc.
                # We always take the 3 corner nodes of each triangle:
                # Gmsh's node ordering for a tri is [v1, v2, v3, (mid-edge nodes ...)]
                if gmsh.model.mesh.getElementProperties(etype)[0].lower().startswith("triangle"):
                    nper = gmsh.model.mesh.getElementProperties(etype)[3]  # nodes per element
                    if nper < 3:
                        continue
                    # walk by nper and take first 3 entries as the triangle's vertices
                    for i in range(0, len(nodes_flat), nper):
                        a, b, c = nodes_flat[i], nodes_flat[i+1], nodes_flat[i+2]
                        triangles.append((int(a), int(b), int(c)))
                        used_nodes.update((int(a), int(b), int(c)))

        for (_, sTag) in surf_ents:
            collect_from_surface(sTag)

        # Fallback: build the boundary ("skin") from tetrahedra, if no 2D tris were found
        if not triangles:
            print("[info] No 2D triangle elements found; extracting boundary from 3D tets...")
            vols = gmsh.model.getEntities(3)
            faces = {}
            for (_, vTag) in vols:
                etypes, etags, enodes = gmsh.model.mesh.getElements(3, vTag)
                for etype, nodes_flat in zip(etypes, enodes):
                    # Linear tetra is type 4; high-order also possible
                    name = gmsh.model.mesh.getElementProperties(etype)[0].lower()
                    if "tetra" not in name:
                        continue
                    nper = gmsh.model.mesh.getElementProperties(etype)[3]
                    for i in range(0, len(nodes_flat), nper):
                        v1, v2, v3, v4 = map(int, nodes_flat[i:i+4])
                        # 4 faces per tet (use sorted tuples so shared faces cancel)
                        for f in ((v1, v2, v3), (v1, v2, v4), (v1, v3, v4), (v2, v3, v4)):
                            key = tuple(sorted(f))
                            faces[key] = faces.get(key, 0) + 1
            # boundary faces appear only once
            for f, cnt in faces.items():
                if cnt == 1:
                    triangles.append(f)
                    used_nodes.update(f)

        if not triangles:
            raise RuntimeError("No surface triangles could be extracted.")

        # Get coordinates for all nodes, then filter to 'used_nodes'
        # nodeTags: [id1, id2, ...]; coords: [x1,y1,z1, x2,y2,z2, ...]
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()
        tag_to_xyz = {}
        for idx, tag in enumerate(nodeTags):
            x, y, z = coords[3*idx : 3*idx+3]
            tag_to_xyz[int(tag)] = (float(x), float(y), float(z))

        # Build a compact 1-based indexing for the used nodes
        used_sorted = sorted(used_nodes)
        tag2new = {tag: i+1 for i, tag in enumerate(used_sorted)}

        # Write .vert
        with open(f"{out_base}.vert", "w", encoding="utf-8") as fv:
            for tag in used_sorted:
                x, y, z = tag_to_xyz[tag]
                fv.write(f"{x}\t{y}\t{z}\n")

        # Write .triv (map original node tags to compact 1-based indices)
        with open(f"{out_base}.triv", "w", encoding="utf-8") as ft:
            for a, b, c in triangles:
                ft.write(f"{tag2new[a]}\t{tag2new[b]}\t{tag2new[c]}\n")

        print(f"Wrote: {out_base}.vert  and  {out_base}.triv")

    finally:
        gmsh.finalize()

if __name__ == "__main__":
    main()
