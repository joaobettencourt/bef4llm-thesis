from ollama import Client
from copy import deepcopy
from pydantic import BaseModel
import os


def get_ollama_host():
    host = os.getenv("OLLAMA_HOST")
    
    if not host:
        raise RuntimeError(
            "OLLAMA_HOST is not set.\n"
            "Make sure you defined it in your .env file and passed it to Docker.\n"
            "Example:\n"
            "  OLLAMA_HOST=http://host.docker.internal:11434"
        )
    
    print(f"[DEBUG] Using OLLAMA_HOST={host}")

    return host

class ConnectLLMs():
    """
    Class to connect to an LLM via Ollama API
    """
    def __init__(self, llm_modell, sys_msg, load_timeout=1800, generate_timeout=300, keep_alive="60m"):
        """
        Initializes the class.

        Uses a single ollama.Client for the entire lifetime of this object. The
        client is first created with a long timeout (load_timeout) so that the
        initial warm-up request (which forces Ollama to load the model into
        memory) doesn't time out prematurely. After warm-up, the client's
        timeout is reconfigured to a shorter value (generate_timeout), so that
        regular generation calls fail fast if something gets stuck, instead of
        waiting up to load_timeout every time.

        Parameters
        ----------
        llm_modell: str
            LLM model (ollama tag is needed)
        sys_msg: str
           System message
        load_timeout: int
            Timeout in seconds used only for the initial warm-up request
            (the one that pays the cost of loading the model into memory).
        generate_timeout: int
            Timeout in seconds used for every chat request after warm-up.
        keep_alive: str
            How long Ollama should keep the model loaded in memory between
            requests (Ollama's own default is "5m"). Passed on every request
            so the model doesn't get unloaded mid-run (e.g. during long RAG /
            validation steps between LLM calls), which would otherwise force
            a reload under the shorter generate_timeout.
        """
        self.llm = llm_modell
        self.chat_histroy = []
        self.sys_msg = sys_msg
        self.keep_alive = keep_alive
        self.generate_timeout = generate_timeout

        # Client is created with the long (load) timeout first.
        self.client = Client(host=get_ollama_host(), timeout=load_timeout)

        # Warm-up: this is the request that actually loads the model into memory.
        self.init_sys_role(sys_msg)
        print(f"LLM {self.llm} loaded. Load timeout was {load_timeout}s, keep_alive={self.keep_alive}")

        # NOTE: ollama.Client doesn't expose a public way to change the timeout
        # after creation. It stores the underlying httpx.Client as the private
        # attribute `_client`, and httpx.Client.timeout is a read/write property,
        # so we reconfigure it directly here. This depends on ollama's internal
        # implementation (tested with ollama~=0.4.4) — if the dependency is
        # upgraded, double check this attribute still exists before relying on it.
        self.client._client.timeout = generate_timeout

        print(f"LLM {self.llm} is ready. Generate timeout is now set to {generate_timeout}s")

    def chat_with_history(self, role:str, content:str, format=None):
        """
        Allows to send a request to the LLM and returns the response.
        The conversation is saved in the history

        Parameters
        ----------
        role: str
            Role of the request
        content: str
            Content of the request
        format: dict
            Desired format of the response - not mandatory

        Returns
        -------
        response: str
            Content string of response from the LLM
        """
        self.chat_histroy.append({"role": "user", "content": content})
        if format:
            response = self.client.chat(model=self.llm,
                                        messages=self.chat_histroy,
                                        format=format,
                                        options={"temperature": 0.1, "num_ctx": 40000},
                                        keep_alive=self.keep_alive)  # numctx = context
        else:
            response = self.client.chat(model=self.llm,
                                    messages=self.chat_histroy,
                                    options={"temperature": 0.1, "num_ctx": 40000},
                                    keep_alive=self.keep_alive) #numctx = context

        self.chat_histroy.append({"role": "assistant", "content": response.message.content})
        return response.message.content

    def chat_without_history(self, role:str, content:str, format=None):
        """
        Allows to send a request to the LLM and returns the response.
        The conversation is NOT saved in the history

        Parameters
        ----------
        role: str
            Role of the request
        content: str
            Content of the request
        format: dict
            Desired format of the response - not mandatory

        Returns
        -------
        response: dict
            Content string of response from the LLM
        """
        # history only consists of the system message
        msg = deepcopy(self.chat_histroy)
        msg.append({"role": role, "content": content})
        if format:
            response = self.client.chat(model=self.llm,
                                        messages=msg,
                                        format=format,
                                        options={"temperature": 0.1, "num_ctx": 40000},
                                        keep_alive=self.keep_alive)
        else:
            response = self.client.chat(model=self.llm,
                                        messages=msg,
                                        options={"temperature": 0.1, "num_ctx": 40000},
                                        keep_alive=self.keep_alive)
        return response.message.content

    def init_sys_role(self, msg:str):
        """
        Allows to change the system message - history is deleted

        Parameters
        ----------
        msg: str
            new system message

       """
        self.chat_histroy = []
        self.chat_histroy.append({"role": "system", "content": msg})
        response = self.client.chat(model=self.llm, messages=self.chat_histroy, keep_alive=self.keep_alive)
        self.chat_histroy.append({"role": "assistant", "content": response.message.content})

    def reset_chat_history(self):
        """
        deletes the chat history, without deleting the system message
        """
        if len(self.chat_histroy) > 1:
            self.chat_histroy = self.chat_histroy[:2]