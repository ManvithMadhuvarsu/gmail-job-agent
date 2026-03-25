"""
daemon.py
Runs the MailAI agent continuously in a loop.
Checks for new emails every POLL_INTERVAL_MINUTES (default: 300 = 5 hours).
"""

import time
import os
import sys
import traceback
from pathlib import Path
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Ensure we can import from main
sys.path.insert(0, str(Path(__file__).parent))

from main import run


def start_daemon():
    load_dotenv()
    init(autoreset=True)
    
    interval_minutes = int(os.getenv("POLL_INTERVAL_MINUTES", "").strip() or 300)
    interval_seconds = interval_minutes * 60
    
    # Display in hours if >= 60 minutes
    if interval_minutes >= 60:
        display_interval = f"{interval_minutes / 60:.1f} hours"
    else:
        display_interval = f"{interval_minutes} minutes"
    
    print(f"{Fore.GREEN}================================================={Style.RESET_ALL}")
    print(f"{Fore.GREEN}🚀 Starting MailAI in 24/7 Daemon Mode...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}⏳ Polling frequency: Every {display_interval}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}================================================={Style.RESET_ALL}")

    consecutive_errors = 0

    while True:
        target_run_time = time.time() + interval_seconds
        
        try:
            run()
            consecutive_errors = 0  # Reset on success
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Daemon gracefully stopped by user.{Style.RESET_ALL}")
            break
        except Exception as e:
            consecutive_errors += 1
            print(f"\n{Fore.RED}💥 Error in daemon cycle #{consecutive_errors}: {e}{Style.RESET_ALL}")
            traceback.print_exc()
            
            if consecutive_errors >= 5:
                # Back off more aggressively if errors keep happening
                backoff = min(consecutive_errors * 60, 3600)  # Max 1 hour backoff
                print(f"{Fore.YELLOW}⏸️  Too many consecutive errors. Backing off for {backoff // 60} minutes...{Style.RESET_ALL}")
                try:
                    time.sleep(backoff)
                except KeyboardInterrupt:
                    break
            else:
                print(f"{Fore.YELLOW}Retrying on next cycle...{Style.RESET_ALL}")
        
        # Calculate how long to sleep (if run() took a long time, we subtract it)
        sleep_time = max(0, target_run_time - time.time())
        next_run = time.strftime('%d %b %Y  %H:%M:%S', time.localtime(time.time() + sleep_time))
        
        if sleep_time >= 3600:
            sleep_display = f"{sleep_time / 3600:.1f} hours"
        elif sleep_time >= 60:
            sleep_display = f"{sleep_time / 60:.0f} minutes"
        else:
            sleep_display = f"{sleep_time:.0f} seconds"
        
        print(f"\n{Fore.CYAN}💤 Sleeping for {sleep_display}... Next mailbox check at {next_run}{Style.RESET_ALL}\n")
        
        try:
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Daemon gracefully stopped by user during sleep.{Style.RESET_ALL}")
            break


if __name__ == "__main__":
    start_daemon()
