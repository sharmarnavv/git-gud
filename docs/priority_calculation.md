# Priority Calculation for Missing Skills

## Overview
Priority for each missing skill is determined through a two-step process that combines **category-based base priority** with **job description frequency analysis**.

## Priority Calculation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Determine Skill Category                                │
│  (via _categorize_skill method)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Get Base Priority from Category                         │
│                                                                   │
│  Category                    Base Priority                       │
│  ─────────────────────────────────────────                       │
│  programming_languages       HIGH                                │
│  frameworks                  HIGH                                │
│  databases                   HIGH                                │
│  cloud_platforms             HIGH                                │
│  tools                       MEDIUM                              │
│  soft_skills                 MEDIUM                              │
│  methodologies               MEDIUM                              │
│  other                       MEDIUM (default)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Analyze Job Description Frequency                       │
│  - Combine all job text:                                         │
│    • skills_required list                                        │
│    • tools_mentioned list                                        │
│    • job description text                                        │
│  - Count how many times skill appears (case-insensitive)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Apply Frequency-Based Priority Adjustment               │
│                                                                   │
│  IF skill_mentions >= 3:                                         │
│      return 'high'                                               │
│                                                                   │
│  ELIF skill_mentions >= 2 AND base_priority != 'low':           │
│      IF base_priority == 'high':                                 │
│          return 'high'                                           │
│      ELSE:                                                       │
│          return 'medium'                                         │
│                                                                   │
│  ELSE:                                                           │
│      return base_priority                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Code Implementation

### Location
`resume_parser/gap_analysis.py` - `SkillsGapAnalyzer._determine_priority()` method (line ~385)

### Full Method

```python
def _determine_priority(self, skill: str, category: str, job: ParsedJobDescription) -> str:
    """Determine priority level for a missing skill."""
    
    # Step 1: Get base priority from category
    base_priority = self._skill_categories.get(category, {}).get('priority', 'medium')
    
    # Step 2: Build complete job text
    job_text = ' '.join([
        ' '.join(job.skills_required),
        ' '.join(job.tools_mentioned),
        str(job.metadata.get('description', ''))
    ]).lower()
    
    # Step 3: Count skill mentions
    skill_mentions = job_text.count(skill.lower())
    
    # Step 4: Adjust priority based on frequency
    if skill_mentions >= 3:
        return 'high'
    elif skill_mentions >= 2 and base_priority != 'low':
        return 'high' if base_priority == 'high' else 'medium'
    
    return base_priority
```

## Priority Levels Explained

### HIGH Priority
- **Technical skills** (programming languages, frameworks, databases, cloud platforms)
- **Any skill mentioned 3+ times** in job description
- Skills critical for the role

### MEDIUM Priority
- **Tools** (git, jenkins, jira, etc.)
- **Soft skills** (leadership, communication, teamwork)
- **Methodologies** (agile, scrum, devops)
- Skills mentioned 2 times (if base is not low)

### LOW Priority
- Skills rarely mentioned or not in predefined categories
- Nice-to-have skills

## Examples

### Example 1: Python (Programming Language)
```
Skill: Python
Category: programming_languages
Base Priority: HIGH (from category)
Job Mentions: 5 times

Calculation:
  skill_mentions (5) >= 3 → return 'high'
  
Final Priority: HIGH
```

### Example 2: React (Framework, Frequently Mentioned)
```
Skill: React
Category: frameworks
Base Priority: HIGH (from category)
Job Mentions: 2 times

Calculation:
  skill_mentions (2) >= 2 AND base_priority != 'low'
  base_priority == 'high' → return 'high'
  
Final Priority: HIGH
```

### Example 3: Git (Tool, Single Mention)
```
Skill: Git
Category: tools
Base Priority: MEDIUM (from category)
Job Mentions: 1 time

Calculation:
  skill_mentions (1) < 2 → return base_priority
  
Final Priority: MEDIUM
```

### Example 4: Jira (Tool, Multiple Mentions)
```
Skill: Jira
Category: tools
Base Priority: MEDIUM (from category)
Job Mentions: 3 times

Calculation:
  skill_mentions (3) >= 3 → return 'high'
  
Final Priority: HIGH (upgraded due to frequency!)
```

### Example 5: Leadership (Soft Skill)
```
Skill: Leadership
Category: soft_skills
Base Priority: MEDIUM (from category)
Job Mentions: 1 time

Calculation:
  skill_mentions (1) < 2 → return base_priority
  
Final Priority: MEDIUM
```

### Example 6: Communication (Soft Skill, Emphasized)
```
Skill: Communication
Category: soft_skills
Base Priority: MEDIUM (from category)
Job Mentions: 4 times

Calculation:
  skill_mentions (4) >= 3 → return 'high'
  
Final Priority: HIGH (upgraded due to emphasis!)
```

## Key Insights

### 1. **Category Sets the Baseline**
- Technical skills start with HIGH priority
- Tools and soft skills start with MEDIUM priority
- This reflects typical hiring priorities

### 2. **Frequency Overrides Category**
- Any skill mentioned 3+ times becomes HIGH priority
- This captures employer emphasis regardless of skill type
- Example: "Communication" mentioned 4 times → HIGH priority

### 3. **Moderate Frequency Upgrades**
- Skills mentioned 2 times get upgraded (unless already low)
- MEDIUM → MEDIUM (stays same)
- HIGH → HIGH (stays same)
- This recognizes repeated emphasis

### 4. **Single Mentions Keep Base Priority**
- Skills mentioned once keep their category's base priority
- Prevents over-prioritizing every mentioned skill

## Priority Impact on Suggestions

Priority directly affects:

1. **Impact Score Calculation**
   - HIGH priority → 0.8 base impact
   - MEDIUM priority → 0.5 base impact
   - LOW priority → 0.2 base impact

2. **Suggestion Ranking**
   - Priority contributes 20% to overall ranking score
   - Higher priority suggestions appear first

3. **User Presentation**
   - HIGH priority suggestions shown as critical
   - MEDIUM priority shown as important
   - LOW priority shown as optional

## Why This Approach Works

1. **Balances Structure and Context**
   - Category provides consistent baseline
   - Frequency captures job-specific emphasis

2. **Prevents Over-Prioritization**
   - Not every mentioned skill becomes high priority
   - Requires 3+ mentions for automatic upgrade

3. **Recognizes Employer Intent**
   - Repeated mentions signal importance
   - Captures what employer truly values

4. **Handles Edge Cases**
   - Soft skills can be high priority if emphasized
   - Technical skills stay high even with single mention
   - Tools can be upgraded if critical to role

## Summary

**Priority = f(Category Base Priority, Job Description Frequency)**

- Start with category-based priority
- Upgrade based on how often skill appears in job description
- Result: Priority that reflects both skill type and employer emphasis
