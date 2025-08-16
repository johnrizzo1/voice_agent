# Voice Interaction Examples and Expected Outputs

## Overview

This guide provides comprehensive examples of voice interactions with the multi-agent system, including expected responses and behavior patterns. These examples demonstrate how the system routes queries, coordinates agents, and provides natural conversational experiences.

## 🗣️ Basic Voice Interactions

### Single-Agent Queries

#### Weather Queries → InformationAgent

**Voice Input**: _"What's the weather like in London?"_

**Expected Flow**:

1. Speech-to-text converts voice to text
2. System identifies weather-related keywords
3. Routes to InformationAgent
4. Agent uses WeatherTool
5. Natural language response generated
6. Text-to-speech converts response to voice

**Expected Response**:

```
"The weather in London is currently 18 degrees Celsius with partly cloudy skies.
There are light winds from the west at 10 kilometers per hour, and the humidity
is at 65 percent. It's a pleasant day with no precipitation expected."
```

**Voice Input**: _"Get weather for New York in Fahrenheit"_

**Expected Response**:

```
"In New York, the current temperature is 72 degrees Fahrenheit. The sky is
mostly sunny with scattered clouds, light winds from the southwest, and
humidity at 58 percent. It's a beautiful day in the city."
```

#### Mathematical Queries → UtilityAgent

**Voice Input**: _"What is 25 times 7 plus 12?"_

**Expected Flow**:

1. System identifies mathematical expression
2. Routes to UtilityAgent
3. Agent uses CalculatorTool
4. Provides calculation with explanation

**Expected Response**:

```
"Let me calculate that for you. 25 times 7 equals 175, and adding 12 gives us 187.
So 25 times 7 plus 12 equals 187."
```

**Voice Input**: _"Calculate compound interest for $10,000 at 5% for 10 years"_

**Expected Response**:

```
"I'll calculate the compound interest for you. With a principal of $10,000 at
5% annual interest compounded annually for 10 years, the final amount would be
$16,288.95. This means you would earn $6,288.95 in interest over the 10-year period."
```

#### File Operations → ProductivityAgent

**Voice Input**: _"List files in the current directory"_

**Expected Flow**:

1. System identifies file operation request
2. Routes to ProductivityAgent
3. Agent uses FileOpsTool
4. Provides organized file listing

**Expected Response**:

```
"Here are the files in the current directory: I found 8 items - 3 Python files
including main.py and setup.py, 2 markdown files including README.md,
1 configuration directory called 'config', 1 examples directory, and 1 docs directory.
The total directory size is approximately 2.5 megabytes."
```

#### General Conversation → GeneralAgent

**Voice Input**: _"Hello, how are you today?"_

**Expected Response**:

```
"Hello! I'm doing well and ready to help you with any tasks you might have.
I can assist you with calculations, weather information, file operations, web searches,
and coordinate complex multi-step workflows. What would you like to work on today?"
```

## 🔄 Multi-Agent Workflows

### Sequential Workflows

#### Weather and File Workflow

**Voice Input**: _"Get weather for Tokyo and save it to weather_report.txt"_

**Expected Flow**:

1. System creates sequential workflow
2. **Step 1**: InformationAgent gets weather data
3. **Step 2**: ProductivityAgent saves data to file
4. Results aggregated and reported

**Expected Response**:

```
"I'll get the weather for Tokyo and save it for you.

First, let me check the current weather: Tokyo is currently experiencing
22 degrees Celsius with light rain showers. The humidity is 78% with
southeast winds at 15 kilometers per hour.

Now I'm saving this information to weather_report.txt... Done!
I've successfully saved the Tokyo weather report to your file.
The file contains the current conditions, temperature, humidity,
wind information, and timestamp."
```

#### Research and Analysis Workflow

**Voice Input**: _"Search for renewable energy trends and calculate growth percentage"_

**Expected Flow**:

1. **Step 1**: InformationAgent searches for renewable energy information
2. **Step 2**: UtilityAgent calculates growth percentages from data
3. **Step 3**: Results combined and presented

**Expected Response**:

```
"I'll research renewable energy trends and analyze the growth data for you.

From my search, I found that renewable energy capacity has grown significantly.
Solar power increased by 22% last year, wind power by 18%, and hydroelectric
by 8%. Overall, renewable energy sources grew by an average of 16% globally.

Based on this data, if we project forward, renewable energy could represent
45% of global energy production by 2030, compared to the current 28%.
This represents a potential 60% increase in renewable energy adoption over the next decade."
```

### Parallel Workflows

#### Multi-Location Weather

**Voice Input**: _"Get weather for London, New York, and Tokyo simultaneously"_

**Expected Flow**:

1. System creates parallel workflow
2. Three InformationAgent instances work simultaneously
3. Results collected and presented together

**Expected Response**:

```
"I'll check the weather for all three cities at once.

Here's the current weather for your requested locations:

London: 18°C, partly cloudy with light winds
New York: 24°C, sunny with clear skies
Tokyo: 22°C, light rain with high humidity

London has the coolest temperature today, while New York has the clearest conditions.
Tokyo is experiencing some rainfall but temperatures remain comfortable across all three cities."
```

### Pipeline Workflows

#### Financial Analysis Pipeline

**Voice Input**: _"Look up Apple stock price, calculate 10% of current value, and save the analysis"_

**Expected Flow**:

1. **Step 1**: InformationAgent searches for Apple stock price
2. **Step 2**: UtilityAgent calculates 10% of the price (using pipeline data)
3. **Step 3**: ProductivityAgent saves the analysis to file

**Expected Response**:

```
"I'll analyze Apple's stock price and perform the calculation for you.

Current Apple stock price is $178.25 per share. Now calculating 10% of this value...
10% of $178.25 equals $17.83.

I'm now saving this analysis to a file. The analysis includes the current stock price,
the 10% calculation, timestamp, and market context.
File saved as 'apple_stock_analysis_[timestamp].txt'.

In summary: Apple stock at $178.25, 10% value is $17.83."
```

### Collaborative Workflows

#### Multi-Perspective Analysis

**Voice Input**: _"Analyze renewable energy from both economic and environmental perspectives"_

**Expected Flow**:

1. Multiple agents collaborate on different aspects
2. InformationAgent researches data
3. UtilityAgent analyzes economic figures
4. GeneralAgent synthesizes environmental impact
5. Results combined into comprehensive analysis

**Expected Response**:

```
"I'll provide a comprehensive analysis of renewable energy from multiple perspectives.

Economic Perspective:
Renewable energy has become increasingly cost-effective. Solar costs have dropped
by 85% over the past decade, making it the cheapest electricity source in many regions.
The industry now employs over 13 million people globally and represents a $1.1 trillion
annual market with 8-10% growth rates.

Environmental Perspective:
Renewable energy sources produce 90% fewer carbon emissions than fossil fuels.
A complete transition could prevent 2-3 degrees of global warming and significantly
reduce air pollution. Wind and solar have minimal water usage compared to traditional
power plants, helping preserve water resources.

Synthesis:
The convergence of economic viability and environmental necessity makes renewable
energy a compelling choice. The industry offers both immediate economic benefits
and long-term environmental protection, creating a strong case for continued investment
and adoption."
```

## 🎯 Advanced Interaction Patterns

### Context Preservation

#### Multi-Turn Conversation

**Turn 1**: _"What's the weather in Paris?"_

**Response 1**:

```
"The weather in Paris is currently 15 degrees Celsius with overcast skies.
There's a light drizzle with winds from the northwest at 12 kilometers per hour.
Humidity is at 72 percent."
```

**Turn 2**: _"Save that weather information to a file"_

**Expected Flow**:

- System preserves context from previous weather query
- Routes to ProductivityAgent with weather context
- Saves the Paris weather data specifically

**Response 2**:

```
"I'll save the Paris weather information for you. Saving the current Paris weather data
- 15°C, overcast with light drizzle, northwest winds at 12 km/h, 72% humidity -
to paris_weather.txt. File saved successfully!"
```

**Turn 3**: _"Now get the weather for London and compare them"_

**Response 3**:

```
"I'll get London's weather and compare it with Paris for you.

London is currently 18°C with partly cloudy skies and light winds.

Comparison:
- London is 3 degrees warmer than Paris (18°C vs 15°C)
- London has clearer conditions (partly cloudy vs overcast with drizzle)
- Both cities have similar light wind conditions
- London has lower humidity at 58% compared to Paris at 72%

Overall, London has more pleasant weather conditions today compared to Paris."
```

### Error Handling and Recovery

#### Agent Unavailable Scenario

**Voice Input**: _"Calculate the square root of 144"_

**Expected Flow** (if UtilityAgent temporarily unavailable):

1. System attempts to route to UtilityAgent
2. Detects agent unavailability
3. Falls back to GeneralAgent with calculator tool
4. Provides response with fallback notice

**Expected Response**:

```
"I'll calculate the square root of 144 for you. The square root of 144 is 12.
This is because 12 multiplied by itself equals 144."
```

#### Tool Failure Recovery

**Voice Input**: _"Get weather for London"_

**Expected Flow** (if WeatherTool fails):

1. InformationAgent attempts weather lookup
2. Tool fails or times out
3. Agent provides helpful error message and alternatives

**Expected Response**:

```
"I apologize, but I'm having trouble accessing weather data right now.
This could be due to a temporary network issue or service maintenance.
You might want to try again in a few minutes, or I can help you with
other tasks like calculations or file operations."
```

### Complex Command Processing

#### Multi-Component Request

**Voice Input**: _"Calculate my monthly budget if I earn $5000 and spend 30% on rent, 20% on food, and 15% on transportation, then save the breakdown"_

**Expected Flow**:

1. UtilityAgent performs multiple calculations
2. GeneralAgent organizes the budget breakdown
3. ProductivityAgent saves the structured data

**Expected Response**:

```
"I'll calculate your monthly budget breakdown and save it for you.

With a monthly income of $5,000:
- Rent (30%): $1,500
- Food (20%): $1,000
- Transportation (15%): $750
- Total allocated: $3,250
- Remaining budget: $1,750 (35%)

This leaves you with $1,750 for other expenses like savings, entertainment,
utilities, and miscellaneous costs.

I'm now saving this budget breakdown to 'monthly_budget_breakdown.txt'
with detailed calculations and percentages. File saved successfully!

Your budget allocation looks balanced with a healthy amount remaining
for other expenses and savings."
```

## 🎤 Voice Command Best Practices

### Clear and Natural Commands

#### Effective Voice Patterns

**✅ Good**: _"What's the weather like in London?"_

- Clear location specified
- Natural conversational tone
- Specific request

**✅ Good**: _"Calculate 15% tip on a $45 restaurant bill"_

- Specific calculation request
- All necessary parameters included
- Context provided

**✅ Good**: _"Get weather for Tokyo and save it to today's weather file"_

- Multi-step request clearly stated
- Both actions specified
- File naming guidance provided

#### Commands to Avoid

**❌ Avoid**: _"Um, like, what's it like outside?"_

- Vague location
- Unclear what information is wanted
- Excessive filler words

**❌ Avoid**: _"Do some math stuff"_

- No specific operation requested
- No parameters provided
- Too vague for processing

**❌ Avoid**: _"Fix my computer"_

- Outside system capabilities
- No specific problem described
- Unrealistic expectation

### Optimal Voice Interaction Tips

#### Speaking Guidelines

1. **Speak Clearly**: Articulate words distinctly
2. **Normal Pace**: Not too fast or too slow
3. **Consistent Volume**: Maintain steady voice level
4. **Pause Appropriately**: Brief pause before and after commands
5. **Natural Tone**: Conversational, not robotic

#### Command Structure

1. **Start with Action**: "Calculate...", "Get...", "Save..."
2. **Provide Context**: Include necessary parameters
3. **Be Specific**: Exact locations, numbers, file names
4. **Use Natural Language**: Speak as you would to a person
5. **One Task per Command**: For simple requests

#### Multi-Step Commands

**Good Structure**: _"Get weather for Paris, then calculate heating costs if temperature is below 10°C, and save the analysis"_

**Benefits**:

- Clear sequence of actions
- Conditional logic specified
- End goal stated
- All context provided

## 🔊 Audio Quality and Recognition

### Optimal Audio Environment

#### Microphone Setup

- **Distance**: 6-12 inches from microphone
- **Angle**: Speak directly toward microphone
- **Background Noise**: Minimize ambient noise
- **Room Acoustics**: Avoid echoes and reverb

#### Environmental Factors

- **Quiet Space**: Reduce background noise
- **Consistent Lighting**: Good for visual feedback
- **Stable Network**: For model downloads and updates
- **Adequate Processing**: Sufficient CPU/RAM available

### Common Audio Issues and Solutions

#### Low Recognition Accuracy

**Problem**: System doesn't understand speech well

**Solutions**:

```yaml
# Adjust VAD settings
audio:
  vad_aggressiveness: 2 # Reduce from 3 to 2
  min_speech_frames: 6 # Reduce sensitivity

# Try different STT model
stt:
  model: "whisper-small" # More accurate than base
```

#### Audio Feedback or Echo

**Problem**: System hears its own voice

**Solutions**:

```yaml
# Increase cooldown periods
audio:
  speech_detection_cooldown: 3.0 # Longer pause after TTS

tts:
  post_tts_cooldown: 1.0 # Wait after speech
```

## 📊 Performance Expectations

### Response Time Guidelines

| Query Type           | Expected Response Time | Notes                        |
| -------------------- | ---------------------- | ---------------------------- |
| Simple Calculation   | 1-3 seconds            | Fast mathematical operations |
| Weather Query        | 2-5 seconds            | Network-dependent            |
| File Operations      | 1-4 seconds            | Depends on file size         |
| Web Search           | 3-8 seconds            | Network and processing time  |
| Multi-Agent Workflow | 5-15 seconds           | Complexity-dependent         |
| Complex Analysis     | 10-30 seconds          | Multi-step processing        |

### Quality Indicators

#### Good Performance Signs

- ✅ Clear, natural voice synthesis
- ✅ Accurate speech recognition
- ✅ Appropriate agent routing
- ✅ Contextual responses
- ✅ Smooth conversation flow

#### Performance Issues

- ❌ Frequent recognition errors
- ❌ Wrong agent routing
- ❌ Slow response times
- ❌ Robotic or unclear speech
- ❌ Context loss between interactions

This comprehensive guide provides realistic expectations for voice interactions with the multi-agent system. Use these examples as a reference for understanding system capabilities, optimizing your voice commands, and troubleshooting interaction issues.
