import logging
import requests
from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class IbmWatsonxModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            response = requests.post(
                "https://iam.cloud.ibm.com/identity/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": credentials.get("ibm_api_key")
                }
            )
            response.raise_for_status()
            return response.json()["access_token"]
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 400:
                logger.warning("Invalid IBM API key or malformed request.")
                raise CredentialsValidateFailedError(
                    "Invalid IBM API key or bad request. Please verify your credentials."
                )
            logger.exception("HTTP error during IBM credential validation.")
            raise CredentialsValidateFailedError(
                f"HTTP error occurred: {http_err}"
            )
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise ex
