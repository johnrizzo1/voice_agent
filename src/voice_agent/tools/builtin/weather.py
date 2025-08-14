"""
Weather tool for getting weather information.
"""

import logging
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from ..base import Tool


class WeatherParameters(BaseModel):
    """Parameters for the weather tool."""

    location: str = Field(description="Location to get weather for")
    units: str = Field(
        default="celsius", description="Temperature units (celsius or fahrenheit)"
    )


class WeatherTool(Tool):
    """
    Weather tool for getting current weather information.

    Uses OpenWeatherMap API for real weather data.
    Falls back to mock data if no API key is configured.
    """

    name = "weather"
    description = "Get current weather information for a location"
    version = "1.0.0"

    Parameters = WeatherParameters

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the weather tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or config."""
        import os

        # Try to get from environment variable
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            self.logger.warning(
                "No OpenWeatherMap API key found. Set OPENWEATHER_API_KEY environment variable for real weather data."
            )
        return api_key

    def execute(self, location: str, units: str = "celsius") -> Dict[str, Any]:
        """
        Get weather information for a location.

        Args:
            location: Location to get weather for
            units: Temperature units (celsius or fahrenheit)

        Returns:
            Dictionary containing weather information
        """
        try:
            if self.api_key:
                # Use real API
                return self._get_real_weather_data(location, units)
            else:
                # Fall back to mock data
                self.logger.info("Using mock weather data - no API key configured")
                mock_weather_data = self._get_mock_weather_data(location, units)
                return {
                    "success": True,
                    "result": mock_weather_data,
                    "error": None,
                    "location": location,
                    "units": units,
                }

        except Exception as e:
            self.logger.error(f"Weather tool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "location": location,
                "units": units,
            }

    def _get_real_weather_data(self, location: str, units: str) -> Dict[str, Any]:
        """Get real weather data from OpenWeatherMap API."""
        try:
            # Convert units for API
            api_units = "metric" if units.lower() == "celsius" else "imperial"
            temp_unit = "°C" if units.lower() == "celsius" else "°F"
            wind_unit = "m/s" if units.lower() == "celsius" else "mph"

            # Make API request
            params = {"q": location, "appid": self.api_key, "units": api_units}

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse response
            weather_data = {
                "location": f"{data['name']}, {data['sys']['country']}",
                "temperature": round(data["main"]["temp"], 1),
                "temperature_unit": temp_unit,
                "condition": data["weather"][0]["main"],
                "weather_description": data["weather"][0]["description"].title(),
                "emoji": self._get_weather_emoji(data["weather"][0]["main"]),
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "wind_unit": wind_unit,
                "wind_direction": data["wind"].get("deg", 0),
                "visibility": data.get("visibility", 0) / 1000,  # Convert to km
                "feels_like": round(data["main"]["feels_like"], 1),
                "min_temp": round(data["main"]["temp_min"], 1),
                "max_temp": round(data["main"]["temp_max"], 1),
                "sunrise": data["sys"]["sunrise"],
                "sunset": data["sys"]["sunset"],
                "timezone": data["timezone"],
                "description": f"Current weather in {data['name']}: {data['weather'][0]['description'].title()}, {round(data['main']['temp'], 1)}{temp_unit}",
                "source": "OpenWeatherMap API",
            }

            return {
                "success": True,
                "result": weather_data,
                "error": None,
                "location": location,
                "units": units,
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Location '{location}' not found",
                    "location": location,
                    "units": units,
                }
            else:
                raise
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "result": None,
                "error": f"Weather API request failed: {str(e)}",
                "location": location,
                "units": units,
            }

    def _get_weather_emoji(self, condition: str) -> str:
        """Get emoji for weather condition."""
        emoji_map = {
            "Clear": "☀️",
            "Clouds": "☁️",
            "Rain": "🌧️",
            "Drizzle": "🌦️",
            "Thunderstorm": "⛈️",
            "Snow": "❄️",
            "Mist": "🌫️",
            "Fog": "🌫️",
            "Haze": "🌫️",
            "Dust": "🌪️",
            "Sand": "🌪️",
            "Ash": "🌋",
            "Squall": "🌪️",
            "Tornado": "🌪️",
        }
        return emoji_map.get(condition, "🌤️")

    def _get_mock_weather_data(self, location: str, units: str) -> Dict[str, Any]:
        """
        Generate mock weather data for demonstration.

        Args:
            location: Location name
            units: Temperature units

        Returns:
            Mock weather data dictionary
        """
        # Convert temperature based on units
        temp_celsius = 22
        if units.lower() == "fahrenheit":
            temperature = (temp_celsius * 9 / 5) + 32
            temp_unit = "°F"
        else:
            temperature = temp_celsius
            temp_unit = "°C"

        # Mock weather conditions based on location
        conditions_map = {
            "london": ("Rainy", "🌧️"),
            "paris": ("Cloudy", "☁️"),
            "tokyo": ("Sunny", "☀️"),
            "new york": ("Partly Cloudy", "⛅"),
            "sydney": ("Sunny", "☀️"),
            "moscow": ("Snowy", "❄️"),
        }

        location_lower = location.lower()
        condition, emoji = conditions_map.get(location_lower, ("Sunny", "☀️"))

        return {
            "location": location.title(),
            "temperature": temperature,
            "temperature_unit": temp_unit,
            "condition": condition,
            "emoji": emoji,
            "humidity": 65,
            "wind_speed": 12,
            "wind_unit": "km/h",
            "description": f"Current weather in {location.title()}: {condition}, {temperature}{temp_unit}",
            "note": "This is mock data for demonstration purposes",
        }

    def get_supported_units(self) -> List[str]:
        """
        Get list of supported temperature units.

        Returns:
            List of supported units
        """
        return ["celsius", "fahrenheit"]

    def get_help(self) -> Dict[str, Any]:
        """
        Get help information for the weather tool.

        Returns:
            Dictionary containing help information
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "location": "Name of the city or location",
                "units": "Temperature units (celsius or fahrenheit)",
            },
            "examples": [
                "Get weather for London",
                "Get weather for New York in fahrenheit",
                "Check temperature in Tokyo",
            ],
            "supported_units": self.get_supported_units(),
            "note": "Uses OpenWeatherMap API when OPENWEATHER_API_KEY environment variable is set",
        }
