"""
test_isolation.py
------------------
Standalone script proving access isolation: for each registered user, run
the same query and show that retrieved chunks only ever come from
companies that user is authorized for. Useful to run live during an
interview demo as objective evidence, instead of eyeballing the UI.

Run with:
    python src/test_isolation.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from access_control import list_registered_users
from qa_engine import get_or_create_session


PROBE_QUERY = "What were the key financial highlights reported?"


def main():
    print("=" * 70)
    print("ACCESS ISOLATION TEST")
    print(f"Probe query (same for every user): \"{PROBE_QUERY}\"")
    print("=" * 70)

    all_passed = True

    for user_email in list_registered_users():
        allowed = config.USER_ACCESS_MAP[user_email]
        print(f"\n--- {user_email}  (authorized: {allowed}) ---")

        session = get_or_create_session(user_email)
        result = session.ask(PROBE_QUERY)

        companies_seen = {s["company"] for s in result["sources"]}
        unauthorized = companies_seen - set(allowed)

        for s in result["sources"]:
            flag = "OK" if s["company"] in allowed else "LEAK!!"
            print(f"  [{flag}] chunk from {s['company']} ({s['source_file']}, p.{s['page']})")

        if unauthorized:
            print(f"  >>> FAIL: retrieved unauthorized companies: {unauthorized}")
            all_passed = False
        else:
            print(f"  >>> PASS: all retrieved chunks within authorized scope.")

    print("\n" + "=" * 70)
    print("RESULT: ALL ISOLATION CHECKS PASSED" if all_passed else "RESULT: ISOLATION FAILURE DETECTED")
    print("=" * 70)


if __name__ == "__main__":
    main()