import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class EmailNotifer:
    """
    Handles email notifications on different events
    """
    def __init__(
            self,
            gmail_address: str,
            app_password: str,
            to_address: str = None
    ):
        self.gmail_address = gmail_address
        self.app_password = app_password
        self.to_address = to_address or gmail_address # Default to myself

    def _send(self, subject: str, body: str):
        """
        Send an email with the provided subject and body
        
        <INPUTS>
        subject: The email subject
        body: The body in the email
        """
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.gmail_address
        msg['To'] = self.to_address
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.gmail_address, self.app_password)
                server.send_message(msg)
            print(f"[EMAIL] Sent: {subject}")
        except Exception as e:
            print(f"[EMAIL] Failed to send '{subject}': {e}")

    def notify_started(self, prefix: str, model: str):
        """
        Structures the email when a model starts generating responses

        <INPUTS>
        prefix: The model prefix generating responses (shorthand version)
        model: The name of the model generating responses
        """
        self._send(
            subject=f"[EXPERIMENT] {prefix} started",
            body=(
                f"Model: {model}\n"
                f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        )

    def notify_completed(self, prefix: str, model: str, started_at: datetime):
        """
        Structures an experiment completion email

        <INPUTS>
        prefix: The model prefix generating responses (shorthand version)
        model: The name of the model generating responses
        started_at: When the experiment started
        """
        elapsed = datetime.now() - started_at
        self._send(
            subject=f"[EXPERIMENT] {prefix} completed",
            body=(
                f"Model: {model}\n"
                f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Elapsed: {str(elapsed).split('.')[0]}"
            )
        )

    def notify_failed(self, prefix: str, model: str, error: Exception):
        self._send(
            subject=f"[EXPERIMENT] {prefix} failed",
            body=(
                f"Model: {model}\n"
                f"Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Error:\n{traceback.format_exc()}"
            )
        )

    def notify_update(self, prefix: str, model: str, scenario: str, lang: str):
        """
        Structures an experiment update email

        <INPUTS>
        prefix: The model prefix generating responses
        model: The name of the model generating responses
        scenario: The scenario started
        lang: The language started
        """
        self._send(
            subject=f"[EXPERIMENT] {prefix} update",
            body=(
                f"Model: {model}\n"
                f"Starting scenario '{scenario}' for language '{lang}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        )