# Introduction
After multiple attempts of building this project through the use of AI tools, and falling into the slippery downfall of leaning completely on those tools to complete the project, and eventually having no idea what is happening, I will take a more hands on approach with this attempt, and build out as much of it as I can, using those tools only for slight assistance with syntax and debugging. 

# Description
The primary goal of this project is pretty straightforward. We want to be able to estimate the number of impressions a given linkedin post gets. 

# Users
The potential users of this would be people in smaller to medium sized companies who want to be able to estimate their total reach to see if they are reaching more people. Although inviduals can use this, based on my own observations, they would get far less benefit out of it as invididuals and influencers are not mentioned as frequently compared to companies. 

# Problem
Linkedin provides impressions information for a company's posts, or when people post as a member of the company, but from a company view, they are unable to track the number of impressions that external content creators get on posts that mention the company. Especially for companies that are trying to see their overall reach, and if they are growing in the public eye, they would want to not only see the posts from their own company, but how many impressions they are getting from external posts. 

# Why
This is important so that people in the company can better understand how their company is getting reach, as well as understand who are the peopple hwo are helping that company get that reach, and if there are collaborative opportunities that exist with those people. 

# Solution / Design
The first part of the solution is to create a simple estimator, such that when a user provides 3 metrics (Number of Reactions, Number of Comments, and Number of Reposts), the model will predict the total number of impressions that that post got. 

The second part of the solution would be to allow users to do a batch estimation. That is, they will be able to insert a spreadsheet of data (formatted in a specific manner), and then the estimator will return a spreadsheet with the estimations filled in. 

I am not too sure how feasible this part is, but I would also like to have a feature so that users can retrain the model by providing a training set where impressions are filled in. The ensemble model can be retrained on that, providing users with more accurate data. 

Lastly, I want to create a dashboard so that users can see information that is relevant to them such as
1. Top posters for the selected time
2. Number of posts and impressions during that time 
3. A time series chart on a monthly basis 
4. Unique number of people who have mentioned the company during that month. 

In order for this to work, the data must be AT LEAST sectioned into the following categories
1. DATE: Preferably, this is in a full date time, although just the month and the year would be fine (08-2025 or 08-01-2025)
2. REACTIONS: an integer. since this is a simple model, we aren't tracking the type of reactions
3. COMMENTS: an integer, just want to see the total number of comments, less concerned with what they say
4. REPOSTS: an integer, just the number of times the post has been reposted. 

# Considerations
Some things to consider with this is that certain types of posts get wildly different impressions. For example, an informative piece may have a very wide reach because a lot of people benefit from the data they are shown, although there may be fewer comments. On the other hand, someone who is posting about a job update may have many comments, but fewer impressions because people do not tend to share those posts to their network. 

Additionally, the types of comments could matter, as someone just saying "good job" doesn't provide room for discussion, while someone asking about methodology could spark a conversation which could generate more views.

Impressions can also differ a significant amount depending on the number of followers a poster has, but again, this is a level of complexity that stretches outside of the bounds of this project since we are also considering the amount of time it takes to collect this information. 


test?