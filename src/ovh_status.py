if __name__ == '__main__':
    import argparse
    import logging
    from colorama import Fore, Back, Style
    from ovh import *

    logger = logging.getLogger("ovh/status")
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    novac = get_nova_client()

    results = {}
    for instance in novac.servers.list():
        ssh_client = get_ssh_client(instance)
        results[instance.name] = {
            "running": is_worker_running(ssh_client),
            "uptime": get_uptime(ssh_client),
        }

    logger.info(Style.RESET_ALL)
    for instance_name in sorted(results):
        uptime = results[instance_name]["uptime"]
        if results[instance_name]["running"]:
            running = (Fore.GREEN + "worker on" + Style.RESET_ALL)
        else:
            running = (Fore.RED + "worker off" + Style.RESET_ALL)
        logger.info(f"{Style.BRIGHT + instance_name + Style.RESET_ALL:>15} {uptime} {running}")
