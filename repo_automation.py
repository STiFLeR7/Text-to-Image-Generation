from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import os


def automate_repo():
    repo_path = "D:/Text-to-Image-Generation"
    os.chdir(repo_path)

    # Step 1: Update README
    with open("README.md", "a") as file:
        file.write(f"\nLast updated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 2: Format code
    os.system("black .")

    # Step 3: Update dependencies
    os.system("pip install --upgrade -r requirements.txt")
    os.system("pip freeze > requirements.txt")

    # Step 4: Commit and push changes
    os.system("git add .")
    commit_message = (
        f"Automated maintenance: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    os.system(f'git commit -m "{commit_message}"')
    os.system("git push origin main")
    print(f"[INFO] Task completed at {datetime.now()}.")


# Create a scheduler
scheduler = BlockingScheduler()

# Schedule the task to run daily at 9:00 AM
scheduler.add_job(automate_repo, "cron", hour=11, minute=46)

print("[INFO] Scheduler started. Waiting for the scheduled time...")
scheduler.start()
