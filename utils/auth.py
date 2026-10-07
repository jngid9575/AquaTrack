import sqlite3
import hashlib


DATABASE_NAME = "aquatrack.db"


def hash_password(password):
    """Convert password into a secure hash."""
    return hashlib.sha256(
        password.encode()
    ).hexdigest()


def create_user(
    name,
    username,
    password,
    role="user"
):
    """Create a new user."""

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()

    password_hash = hash_password(
        password
    )

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                name,
                username,
                password_hash,
                role
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                username,
                password_hash,
                role
            )
        )

        connection.commit()

        success = True

    except sqlite3.IntegrityError:

        success = False

    connection.close()

    return success


def authenticate_user(
    username,
    password
):
    """Check username and password."""

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()

    password_hash = hash_password(
        password
    )

    cursor.execute(
        """
        SELECT
            user_id,
            name,
            username,
            role
        FROM users
        WHERE username = ?
        AND password_hash = ?
        """,
        (
            username,
            password_hash
        )
    )

    user = cursor.fetchone()

    connection.close()

    return user