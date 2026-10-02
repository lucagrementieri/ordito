import warp as wp

from ordito.kernels.array import declare_map_signatures, map_probe, map_probe_single


@wp.func
def first_hit(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> tuple[wp.int32, wp.vec3]:
    # First-hit face index (-1 on miss) and location; ``direction`` need not be unit length.
    #
    # The face index carries the hit test -- both consumers already read it that way
    # (``first_hit_append``'s ``face >= 0`` and ``ray.intersects_first``'s documented ``-1``
    # sentinel) -- so there is no separate hit flag to keep in step with it. On a miss the
    # location is the zero vector and no caller reads it.
    unit_direction = wp.normalize(direction)
    query = wp.mesh_query_ray(mesh_id, origin, unit_direction, max_t)
    if query.result:
        return query.face, origin + unit_direction * query.t
    return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def first_hit_append(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.float32,
    out_index_ray: wp.array[wp.int32],
    out_index_tri: wp.array[wp.int32],
    out_locations: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    # Single-pass compaction: each ray that hits atomically claims one output slot.
    i = wp.int32(wp.tid())
    face, location = first_hit(mesh_id, ray_origins[i], ray_directions[i], max_t)
    if face >= 0:
        slot = wp.atomic_add(out_count, 0, 1)
        out_index_ray[slot] = i
        out_index_tri[slot] = face
        out_locations[slot] = location


@wp.func
def any_hit(mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32) -> wp.bool:
    return wp.mesh_query_ray_anyhit(mesh_id, origin, wp.normalize(direction), max_t)


@wp.func
def longest_ray_distance(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    max_t: wp.float32,
    planar_tol: wp.float32,
) -> wp.float32:
    # ``direction`` need not be unit length; the offset walk below assumes a unit ray.
    #
    # The 64-step cap bounds how many hits inside ``planar_tol`` of the origin the walk will step
    # over before giving up, and exhausting it returns ``wp.inf`` -- i.e. reports the ray as
    # unobstructed when it is not. Reaching that needs 64 surfaces stacked within one
    # ``planar_tol`` band along one ray, which takes coincident or duplicated geometry rather
    # than a fine mesh; a caller who has that should dedupe (``repair.merge_vertices``) instead of
    # raising the cap, since every extra step is a full BVH descent for every ray in the launch.
    unit_direction = wp.normalize(direction)
    t_offset = wp.float32(0.0)
    cur_origin = origin
    for _i in range(64):
        remaining = max_t - t_offset
        if remaining <= wp.float32(0.0):
            break
        query = wp.mesh_query_ray(mesh_id, cur_origin, unit_direction, remaining)
        if not query.result:
            break
        dist = t_offset + query.t
        if dist > planar_tol:
            return dist
        t_offset = t_offset + query.t + planar_tol
        cur_origin = origin + unit_direction * t_offset
        if t_offset >= max_t:
            break
    return wp.inf


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (any_hit, (wp.uint64(1), dense(wp.vec3), dense(wp.vec3), wp.float32(1)), wp.bool),
            (any_hit, (wp.uint64(1), single(wp.vec3), single(wp.vec3), wp.float32(1)), wp.bool),
        ]
    )


_declare_map_kernels()
