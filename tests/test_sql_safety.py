from app.utils.sql_safety import is_safe_sql


def test_is_safe_sql_allows_select():
    assert is_safe_sql("SELECT * FROM users")
    assert is_safe_sql("WITH t AS (SELECT 1) SELECT * FROM t")


def test_is_safe_sql_blocks_non_select():
    assert not is_safe_sql("DELETE FROM users")
    assert not is_safe_sql("UPDATE users SET name='x'")
    assert not is_safe_sql("INSERT INTO users VALUES (1)")


def test_is_safe_sql_blocks_multiple_statements():
    assert not is_safe_sql("SELECT 1; SELECT 2")
