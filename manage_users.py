"""CLI tool to manage users for the Catalan Lecture Processor.

Usage:
    python manage_users.py add <username> <display_name> --password <password>
    python manage_users.py remove <username>
    python manage_users.py list
    python manage_users.py reset <username> --password <new_password>

Examples:
    python manage_users.py add joana.buisel "Joana Buisel" --password abc123
    python manage_users.py add marc.garcia "Marc Garcia" --password xyz789
    python manage_users.py list
    python manage_users.py remove joana.buisel
    python manage_users.py reset joana.buisel --password newpass456
"""

import argparse
import sys
import os
import getpass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.auth import (
    FirebaseAuthError,
    add_user,
    list_users,
    remove_user,
    reset_password,
)


def cmd_add(args):
    password = args.password or getpass.getpass(f"Password for {args.username}: ")
    try:
        if add_user(args.username, password, args.name):
            print(f"  Created user: {args.username} ({args.name})")
        else:
            print(f"  Error: user '{args.username}' already exists")
            sys.exit(1)
    except FirebaseAuthError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)


def cmd_remove(args):
    try:
        if remove_user(args.username):
            print(f"  Removed user: {args.username}")
        else:
            print(f"  Error: user '{args.username}' not found")
            sys.exit(1)
    except FirebaseAuthError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)


def cmd_list(args):
    try:
        users = list_users()
    except FirebaseAuthError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)
    if not users:
        print("  No users registered.")
        return
    print(f"  {'Username':<25} {'Name'}")
    print(f"  {'-'*25} {'-'*30}")
    for u in users:
        print(f"  {u['username']:<25} {u['name']}")
    print(f"\n  Total: {len(users)} user(s)")


def cmd_reset(args):
    password = args.password or getpass.getpass(f"New password for {args.username}: ")
    try:
        if reset_password(args.username, password):
            print(f"  Password reset for: {args.username}")
        else:
            print(f"  Error: user '{args.username}' not found")
            sys.exit(1)
    except FirebaseAuthError as exc:
        print(f"  Error: {exc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage users for Catalan Lecture Processor"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a new user")
    p_add.add_argument("username", help="Login username (e.g. joana.buisel)")
    p_add.add_argument("name", help="Display name (e.g. 'Joana Buisel')")
    p_add.add_argument("--password", "-p", help="Password (prompted if omitted)")
    p_add.set_defaults(func=cmd_add)

    # remove
    p_rm = sub.add_parser("remove", help="Remove a user")
    p_rm.add_argument("username")
    p_rm.set_defaults(func=cmd_remove)

    # list
    p_ls = sub.add_parser("list", help="List all users")
    p_ls.set_defaults(func=cmd_list)

    # reset
    p_reset = sub.add_parser("reset", help="Reset a user's password")
    p_reset.add_argument("username")
    p_reset.add_argument("--password", "-p", help="New password (prompted if omitted)")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
