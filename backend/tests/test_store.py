"""Persistence, migrations, vector-ID alignment and rollback."""

import numpy as np
import pytest

from mediaindex.db import MIGRATIONS, Database
from mediaindex.model.profiles import IndexProfile
from mediaindex.store import INDEXED, PENDING, Store


def unit(seed, dim=768):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    st = Store(db)
    prof = IndexProfile()
    st.register_profile(prof)
    lib = st.create_library("lib", tmp_path)
    return tmp_path, db, st, prof, lib


def add(st, lib, name, h):
    aid, _ = st.upsert_asset(lib["id"], name, "image", size=1, mtime_ns=1, content_hash=h)
    return aid


def test_migration_versions_and_idempotent(tmp_path):
    db = Database(tmp_path / "m.sqlite3")
    assert db.version == len(MIGRATIONS)
    db.close()
    db2 = Database(tmp_path / "m.sqlite3")  # re-running migrate is a no-op
    assert db2.version == len(MIGRATIONS)


def test_identity_survives_restart(env):
    tmp, db, st, prof, lib = env
    ids = {add(st, lib, f"img{i}.jpg", f"h{i}"): i for i in range(5)}
    for aid, i in ids.items():
        st.write_embeddings(aid, prof.key, "image", unit(i), source_hash=f"h{i}")
    db.close()

    st2 = Store(Database(tmp / "t.sqlite3"))
    m = st2.load_matrix(prof.key, [lib["id"]])
    assert len(m.asset_ids) == 5
    for aid, row in zip(m.asset_ids, m.matrix):
        assert np.allclose(row, unit(ids[aid]))  # each vector still maps to its own asset
    # immutable ID for same path on re-import
    aid_again, changed = st2.upsert_asset(lib["id"], "img0.jpg", "image", size=1, mtime_ns=2, content_hash="h0")
    assert ids[aid_again] == 0 and not changed


def test_vector_id_mapping_after_deletes_and_out_of_order_writes(env):
    _, _, st, prof, lib = env
    aids = [add(st, lib, f"f{i}.jpg", f"h{i}") for i in range(10)]
    for i in [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]:
        st.write_embeddings(aids[i], prof.key, "image", unit(i))
    st.remove_asset(aids[3])
    st.remove_asset(aids[8])
    m = st.load_matrix(prof.key)
    assert set(m.asset_ids) == set(aids) - {aids[3], aids[8]}
    for aid, row in zip(m.asset_ids, m.matrix):
        assert np.allclose(row, unit(aids.index(aid)))


def test_partial_write_rolls_back_and_is_not_searchable(env):
    _, _, st, prof, lib = env
    aid = add(st, lib, "a.jpg", "ha")
    bad = np.stack([unit(1), np.full(768, np.nan, np.float32)])
    with pytest.raises(ValueError):
        st.write_embeddings(aid, prof.key, "video-visual", bad, segments=[(0, 8), (4, 12)])
    assert st.get_asset(aid)["status"] == PENDING
    assert st.load_matrix(prof.key).asset_ids == []

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with st.db.tx() as c:
            st.write_embeddings(aid, prof.key, "image", unit(1), c=c)
            raise Boom()
    assert st.get_asset(aid)["status"] == PENDING
    assert st.get_asset_vectors(aid, prof.key) is None


def test_pending_assets_excluded_and_profile_filtered(env):
    _, _, st, prof, lib = env
    a = add(st, lib, "a.jpg", "ha")
    b = add(st, lib, "b.jpg", "hb")
    st.write_embeddings(a, prof.key, "image", unit(1))
    other = IndexProfile(precision="float32")
    st.register_profile(other)
    st.write_embeddings(b, other.key, "image", unit(2))
    assert st.load_matrix(prof.key).asset_ids == [a]
    assert st.load_matrix(other.key).asset_ids == [b]


def test_content_change_invalidates_vectors_and_bumps_generation(env):
    _, _, st, prof, lib = env
    a = add(st, lib, "a.jpg", "h1")
    st.write_embeddings(a, prof.key, "image", unit(1))
    g0 = st.load_matrix(prof.key).generation
    aid, changed = st.upsert_asset(lib["id"], "a.jpg", "image", size=2, mtime_ns=2, content_hash="h2")
    assert aid == a and changed
    assert st.get_asset(a)["status"] == PENDING
    m = st.load_matrix(prof.key)
    assert m.asset_ids == [] and m.generation > g0


def test_duplicate_content_keeps_separate_assets_and_reuses_vectors(env):
    _, _, st, prof, lib = env
    a = add(st, lib, "x/a.jpg", "same")
    b = add(st, lib, "y/a-copy.jpg", "same")
    assert a != b
    st.write_embeddings(a, prof.key, "image", unit(5), source_hash="same")
    reused = st.reusable_vectors("same", prof.key, "image")
    assert reused is not None and np.allclose(reused[0], unit(5))
    st.write_embeddings(b, prof.key, "image", reused, source_hash="same")
    assert sorted(st.load_matrix(prof.key).asset_ids) == sorted([a, b])
    assert len(st.find_by_hash("same")) == 2


def test_remove_asset_keeps_original_file(env):
    tmp, _, st, prof, lib = env
    f = tmp / "keep.jpg"
    f.write_bytes(b"original")
    a = add(st, lib, "keep.jpg", "hk")
    st.remove_asset(a)
    st.delete_library(lib["id"])
    assert f.read_bytes() == b"original"


def test_interrupted_jobs_recovered(env):
    tmp, db, st, _, lib = env
    st.save_job({"id": "j1", "kind": "import", "status": "running", "done": 3, "total": 10}, lib["id"])
    st.save_job({"id": "j2", "kind": "import", "status": "done", "done": 10, "total": 10}, lib["id"])
    db.close()
    st2 = Store(Database(tmp / "t.sqlite3"))
    assert st2.recover_interrupted_jobs() == 1
    assert st2.get_job("j1")["status"] == "interrupted"
    assert st2.get_job("j2")["status"] == "done"
