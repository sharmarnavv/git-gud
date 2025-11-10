# Impact Score Calculation for Missing Skills

## Overview
The impact score for each missing skill is calculated through a multi-step process that considers skill category, priority, and job context.

## Calculation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gap Analysis Process                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Identify Missing Skills                                │
│  - Compare job requirements vs resume skills                     │
│  - Filter out skills with alternatives/synonyms                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Categorize Skill                                        │
│  - programming_languages, frameworks, databases,                 │
│    cloud_platforms, tools, soft_skills, methodologies            │
│  - Based on keyword matching and ontology lookup                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Determine Priority                                      │
│  - Base priority from category (high/medium/low)                 │
│  - Adjusted by frequency in job description:                     │
│    • 3+ mentions → high priority                                 │
│    • 2+ mentions → upgrade by one level                          │
│    • 1 mention → keep base priority                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Calculate Impact Score                                  │
│                                                                   │
│  Formula: impact_score = min(1.0, base_impact × multiplier)     │
│                                                                   │
│  Base Impact (from priority):                                    │
│    • high:   0.8                                                 │
│    • medium: 0.5                                                 │
│    • low:    0.2                                                 │
│                                                                   │
│  Category Multiplier:                                            │
│    • programming_languages: 1.2                                  │
│    • frameworks:           1.1                                   │
│    • databases:            1.1                                   │
│    • cloud_platforms:      1.0                                   │
│    • methodologies:        0.9                                   │
│    • tools:                0.8                                   │
│    • soft_skills:          0.7                                   │
│    • other:                1.0                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Create SkillGap Object                                  │
│  - skill, category, priority, confidence, impact_score           │
│  - alternatives, learning_resources                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: SuggestionEngine Uses Impact Score                      │
│  - Directly uses SkillGap.impact_score                           │
│  - Combines with feasibility_score for ranking                   │
│  - Ranking formula:                                              │
│    score = (impact × 0.4) + (feasibility × 0.3) +               │
│            (priority_weight × 0.2) + (user_pref × 0.1)          │
└─────────────────────────────────────────────────────────────────┘
```

## Examples

### Example 1: High-Priority Programming Language
```
Skill: Python
Category: programming_languages
Job mentions: 5 times
Priority: high (due to 5 mentions)

Calculation:
  base_impact = 0.8 (high priority)
  multiplier = 1.2 (programming_languages)
  impact_score = min(1.0, 0.8 × 1.2) = 0.96
```

### Example 2: Medium-Priority Framework
```
Skill: React
Category: frameworks
Job mentions: 2 times
Priority: high (upgraded from medium due to 2+ mentions)

Calculation:
  base_impact = 0.8 (high priority)
  multiplier = 1.1 (frameworks)
  impact_score = min(1.0, 0.8 × 1.1) = 0.88
```

### Example 3: Low-Priority Tool
```
Skill: Jira
Category: tools
Job mentions: 1 time
Priority: medium (base priority for tools)

Calculation:
  base_impact = 0.5 (medium priority)
  multiplier = 0.8 (tools)
  impact_score = min(1.0, 0.5 × 0.8) = 0.40
```

### Example 4: Soft Skill
```
Skill: Leadership
Category: soft_skills
Job mentions: 1 time
Priority: medium (base priority)

Calculation:
  base_impact = 0.5 (medium priority)
  multiplier = 0.7 (soft_skills)
  impact_score = min(1.0, 0.5 × 0.7) = 0.35
```

## Key Factors Influencing Impact

1. **Skill Category** (Most Important)
   - Technical skills (programming, frameworks, databases) get higher multipliers
   - Soft skills and tools get lower multipliers
   - Reflects typical hiring priorities

2. **Priority Level** (Very Important)
   - Determined by category base + job description frequency
   - More mentions = higher priority = higher impact
   - Reflects employer emphasis

3. **Job Context** (Important)
   - Skills mentioned multiple times are weighted higher
   - Reflects what the employer values most

4. **Category Multiplier** (Moderate)
   - Fine-tunes impact based on skill type
   - Ensures technical skills are prioritized appropriately

## Impact Score Range

- **0.7 - 1.0**: Critical skills (high-priority technical skills)
- **0.5 - 0.7**: Important skills (medium-priority technical or high-priority soft skills)
- **0.3 - 0.5**: Moderate skills (low-priority technical or medium-priority soft skills)
- **0.0 - 0.3**: Nice-to-have skills (low-priority soft skills or tools)

## Usage in SuggestionEngine

The SuggestionEngine receives these pre-calculated impact scores and:

1. **Uses them directly** for skill-based suggestions
2. **Combines with feasibility** to rank suggestions
3. **Applies personalization** to boost scores for user focus areas
4. **Calculates improvement potential** based on top suggestions' impact scores

The impact score represents the **expected improvement to the overall match score** if the candidate acquires that skill.
