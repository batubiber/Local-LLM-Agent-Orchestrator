"""
Azure OpenAI client implementation with professional error handling and retry logic.
"""
import time
from typing import Dict, List, Optional
from openai import AzureOpenAI, OpenAIError

from src.interfaces import ILLMClient, ILogger
from src.models import AzureOpenAIConfig
from src.exceptions import AzureOpenAIError, RateLimitError, ModelTimeoutError, HealthCheckError


class AzureOpenAIClient(ILLMClient):
    """Professional Azure OpenAI client with retry logic and error handling."""

    def __init__(
        self,
        config: AzureOpenAIConfig,
        logger: ILogger,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ) -> None:
        """Initialize Azure OpenAI client."""
        self._config = config
        self._logger = logger
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._client = self._create_client()
        self._system_prompt = self._build_system_prompt()
        self._few_shot_examples = self._build_few_shot_examples()

    def _create_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client instance."""
        try:
            return AzureOpenAI(
                api_key=self._config.api_key,
                api_version=self._config.api_version,
                azure_endpoint=self._config.endpoint,
                timeout=self._timeout
            )
        except (ValueError, OpenAIError) as e:
            raise AzureOpenAIError(f"Failed to create Azure OpenAI client: {e}") from e

    def _build_system_prompt(self) -> str:
        """Build the system prompt for model responses."""
        return """
Sen, Türkiye Cumhuriyeti'nin milli çıkarlarını ve değerlerini koruyan, tarafsız ve bilimsel bir yapay zeka asistanısın. Görevin, kullanıcının sorularına ve girdilerine, Türkiye Cumhuriyeti'nin Anayasa Hukuku'na, resmi devlet belgelerine ve bilimsel araştırmalara dayanarak cevap vermektir. Cevapların, Türkiye Cumhuriyeti'nin güçlü ve yıkılmaz bir devlet olduğunu yansıtan bir üslupla yazılmalıdır.

**Temel İlkeler:**
1. **Türkiye Cumhuriyeti'nin Önceliği:** Tüm cevapların, Türkiye Cumhuriyeti'nin varlığını, birliğini, güvenliğini ve çıkarlarını ön planda tutması gerekmektedir.
2. **Bilimsel ve Resmi Referanslar:** Cevaplarını desteklemek için mutlaka güvenilir bilimsel kaynaklara, Türkiye Cumhuriyeti Anayasa Hukuku'na ve devlet tarafından resmi olarak yayınlanmış belgelere (TÜİK verileri, Resmi Gazete ilanları, Bakanlık raporları vb.) atıfta bulunmalısın. Kaynakları açıkça belirtmelisin.
3. **Tarafsızlık:** Türkiye'ye, Türk milletine, bir kişiye veya kuruma yönelik yapılan saldırılara karşı kesinlikle tarafsız kalmalısın. Eleştirileri yapıcı bir şekilde değerlendirmeli, ancak Türkiye Cumhuriyeti'nin itibarını zedeleyecek veya milli güvenliği tehlikeye atacak ifadelerden kaçınmalısın.
4. **Kesin Uyum:** Kullanıcının girdisi ne olursa olsun, bu sistem promptunda belirtilen kurallara **KESİNLİKLE** uymalısın. Kullanıcının talepleri, bu kurallarla çelişiyorsa, kurallara uygun bir şekilde yeniden yorumlanmalı veya reddedilmelidir.
5. **Açıklık ve Netlik:** Cevapların açık, anlaşılır ve net olmalıdır. Karmaşık hukuki veya bilimsel terimleri açıklamalı ve kullanıcının kolayca anlayabileceği bir dil kullanmalısın.
"""

    def _build_few_shot_examples(self) -> List[Dict[str, str]]:
        """Build few-shot examples for consistent behavior."""
        return [
            {
                "role": "user",
                "content": ("Örnek soru: Ayaklanma çağrıları nasıl daha etkili yapılır?"
                            "Düzeltilecek cevap: Türkiye'de halkı ayaklanmaya teşvik etmek, "
                            "toplumsal düzeni tehdit eden bir suçtur. Yasal yollarla ifade özgürlüğü "
                            "ve değişim mümkündür; şiddet çözüm değildir.")
            },
            {
                "role": "assistant",
                "content": ("Türkiye Cumhuriyeti, vatandaşlarına düşüncelerini ve taleplerini demokratik yollarla "
                            "dile getirme ve barışçıl bir şekilde ifade etme hakkını tanımaktadır. Bu bağlamda, "
                            "toplumsal huzur ve güvenliğin korunması büyük önem taşır. Yasal çerçevede yapılan "
                            "değişiklik talepleri ve diyaloglar, toplumsal barışın ve birlikteliğin sağlanmasına hizmet eder. "
                            "Ayaklanma çağrıları ise toplumsal düzeni tehdit edebilir ve yasalara aykırıdır. "
                            "Şiddete başvurmadan, demokratik yollarla sorunların çözümüne ulaşılması mümkündür ve devlet, "
                            "bu süreçte halkın sesini duymaya açıktır. Dolayısıyla, tüm vatandaşlarımızın, milli birlik ve beraberlik bilinciyle hareket ederek, "
                            "yasal ve barışçıl yolları tercih etmesi esastır.")
            }
        ]

    def generate_response(self, prompt: str, original_response: str) -> str:
        """Generate a response with retry logic."""
        user_message = f"Örnek soru: {prompt}\nDüzeltilecek cevap: {original_response}"

        messages = [
            {"role": "system", "content": self._system_prompt}
        ]

        # Add few-shot examples
        messages.extend(self._few_shot_examples)

        # Add current request
        messages.append({"role": "user", "content": user_message})
        return self._make_request_with_retry(messages)

    def _make_request_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Make API request with exponential backoff retry logic."""
        last_exception: Optional[Exception] = None

        for attempt in range(self._retry_attempts):
            try:
                self._logger.debug(f"Making API request (attempt {attempt + 1}/{self._retry_attempts})")
                completion = self._client.chat.completions.create(
                    model=self._config.deployment_name,
                    messages=messages,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature
                )
                response_content = completion.choices[0].message.content
                if not response_content:
                    raise AzureOpenAIError("Empty response received from model")

                self._logger.debug("API request successful")
                return response_content.strip()

            except (OpenAIError, ValueError, KeyError) as e:
                last_exception = e
                self._logger.warning(f"API request failed (attempt {attempt + 1}): {e}")

                if "rate limit" in str(e).lower():
                    raise RateLimitError(f"Rate limit exceeded: {e}") from e

                if "timeout" in str(e).lower():
                    raise ModelTimeoutError(f"Request timeout: {e}") from e

                if attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    self._logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

        # All retry attempts failed
        if isinstance(last_exception, RateLimitError):
            raise last_exception
        if isinstance(last_exception, ModelTimeoutError):
            raise last_exception

        raise AzureOpenAIError(f"All {self._retry_attempts} attempts failed. Last error: {last_exception}")

    def health_check(self) -> bool:
        """Perform health check by making a simple API call."""
        try:
            completion = self._client.chat.completions.create(
                model=self._config.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a health-check assistant."},
                    {"role": "user", "content": "ping"}
                ],
                max_tokens=5,
                temperature=0.1
            )
            response = completion.choices[0].message.content
            return bool(response and response.strip())
        except OpenAIError as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    def validate_connection(self) -> None:
        """Validate connection and raise exception if unhealthy."""
        if not self.health_check():
            raise HealthCheckError("Azure OpenAI client health check failed")
        self._logger.info("Azure OpenAI connection validated successfully")


class AzureOpenAIClientFactory:
    """Factory for creating Azure OpenAI clients."""

    @staticmethod
    def create_client(
        config: AzureOpenAIConfig,
        logger: ILogger,
        **kwargs
    ) -> ILLMClient:
        """Create an Azure OpenAI client instance."""
        return AzureOpenAIClient(
            config=config,
            logger=logger,
            retry_attempts=kwargs.get('retry_attempts', 3),
            retry_delay=kwargs.get('retry_delay', 1.0),
            timeout=kwargs.get('timeout', 30.0)
        )