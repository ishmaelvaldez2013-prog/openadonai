import os
import rumps
import subprocess

APP_NAME = "OpenAdonAI"
PROJECT_ROOT = "/Users/ishmael/Developer/OpenAdonAI/Tools/rag_service"
OPENADONAI_CMD = "/usr/local/bin/openadonai"


def run_cmd(*args):
    subprocess.Popen([OPENADONAI_CMD, *args])


class OpenAdonAITray(rumps.App):
    def __init__(self):
        super(OpenAdonAITray, self).__init__(APP_NAME)
        self.menu = [
            "Start Oracle",
            "Stop Oracle",
            "Restart Oracle",
            "Doctor",
            "Logs",
            None,
            "Quit",
        ]

    @rumps.clicked("Start Oracle")
    def start_oracle(self, _):
        run_cmd("start")

    @rumps.clicked("Stop Oracle")
    def stop_oracle(self, _):
        run_cmd("stop")

    @rumps.clicked("Restart Oracle")
    def restart_oracle(self, _):
        run_cmd("restart")

    @rumps.clicked("Doctor")
    def doctor(self, _):
        run_cmd("doctor")

    @rumps.clicked("Logs")
    def logs(self, _):
        run_cmd("logs")

    @rumps.clicked("Quit")
    def quit_app(self, _):
        rumps.quit_application()


if __name__ == "__main__":
    OpenAdonAITray().run()