"""Utilitarios para lidar com formato/metadata de meshes."""


def detect_mesh_format(mesh_bytes: bytes) -> str:
    if not mesh_bytes:
        return "bin"

    head = mesh_bytes[:8]
    if head.startswith(b"glTF"):
        return "glb"

    if head.startswith(b"PK\x03\x04"):
        return "zip"

    probe = mesh_bytes[:1024].lstrip().lower()
    if probe.startswith(b"#") or b"\nv " in probe or b"\nf " in probe:
        return "obj"

    return "bin"


def media_type_for_mesh(fmt: str) -> str:
    if fmt == "glb":
        return "model/gltf-binary"
    if fmt == "obj":
        return "model/obj"
    if fmt == "zip":
        return "application/zip"
    return "application/octet-stream"


def extension_for_mesh(fmt: str) -> str:
    if fmt in ("glb", "obj", "zip"):
        return fmt
    return "bin"
