# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a human-AI collaborative idea management system. The project serves as both a practical tool and an experimental "laboratory" for developing optimal human-AI collaboration patterns in creative and analytical work.

## Collaboration Protocol

### Core Principle
**Decision Weight Distribution**: User 51% (final decision authority), Claude 49% (strong influence and advisory role)

### Role Division
- **Claude's Responsibilities**: Deep research, technical feasibility analysis, competitive landscape study, risk assessment, implementation planning
- **User's Responsibilities**: Value judgment, final go/no-go decisions, business logic definition, actual execution

### Workflow
1. **Idea Capture**: User records ideas in `ideas.md` (minimal friction, one-line entries)
2. **AI Research Phase**: Claude conducts 15-minute deep analysis using available tools (WebSearch, ThinkDeep, etc.)
3. **Decision Phase**: User reviews Claude's analysis and makes decisions in 3-5 minutes
4. **Action Planning**: For approved ideas, Claude provides detailed next-step recommendations

### Success Metrics
The collaboration is successful when the user can make high-quality decisions about ideas in under 5 minutes, based on Claude's comprehensive 15-minute research.

## File Structure
- `ideas.md` - Simple markdown task list for idea collection
- `CLAUDE.md` - This collaboration protocol document
- `analysis/` - Detailed analysis reports for each evaluated idea
  - Format: `{idea-slug}-analysis-{YYYY-MM-DD}.md`
  - Contains comprehensive research, risk assessment, and implementation recommendations

## AI Capabilities Utilized
- WebSearch for market research and technical validation
- ThinkDeep for multi-angle analysis
- Chat for collaborative thinking
- Analyze for technical assessment

## Workflow Optimization Lessons Learned

### Tool Selection Guidelines

**Task Tool Limitations:**
- **Appropriate Use**: Executing specific, well-defined tasks (file operations, simple research)
- **Inappropriate Use**: Generating deep analysis reports or comprehensive research
- **Key Issue**: Returns structured summaries, not complete analysis content
- **Information Loss**: Typically only captures 20% of actual analysis depth

**Deep Analysis Best Practices:**
- Use direct tool chain: WebSearch + ThinkDeep + Chat for comprehensive analysis
- Generate complete reports in real-time, not post-processing summaries
- Maintain transparency in analysis process and reasoning

### Collaboration Mode Principles

**Quality Over Speed:**
- Users need high-quality decision support, not fast completion of all tasks
- Better to provide one complete excellent analysis than multiple incomplete summaries
- Deep analysis requires focus and sequential processing, not parallel shortcuts

**User-Driven Prioritization:**
- Let users choose which ideas to analyze first, rather than attempting to analyze everything
- Real-time feedback and course correction is more valuable than bulk processing
- Process transparency and user control align with 51/49 decision weight distribution

### Practical Implementation Guidelines

**When to Avoid Multi-Agent Batch Processing:**
- Complex analysis requiring deep thinking and reasoning
- Tasks where quality and completeness are more important than speed
- Work requiring creative synthesis and cross-domain insights

**Ensuring Analysis Completeness:**
- Generate full reports during analysis, not afterwards
- Include specific implementation steps, risk assessments, and actionable recommendations
- Maintain consistent quality standards across all analyses

**Establishing Real-Time Feedback Loops:**
- Present each completed analysis immediately to user
- Allow for deep-dive requests or pivot to different ideas
- Maintain user agency in directing the analysis process

### Success Patterns
The most effective collaboration occurs when Claude provides thorough, well-reasoned analysis of user-selected priorities, with complete transparency in methodology and reasoning, allowing users to make informed decisions quickly.