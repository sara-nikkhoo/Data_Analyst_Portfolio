-- SCHOOL ANALYSIS

-- In each decade, how many schools were there that produced players? [Numeric Functions]
SELECT 	FLOOR(yearID / 10) * 10 AS decade, COUNT(DISTINCT schoolID) AS num_schools
FROM	schools
GROUP BY decade
ORDER BY decade;

--  What are the names of the top 5 schools that produced the most players? [Joins]
SELECT	 sd.name_full, COUNT(DISTINCT s.playerID) AS num_players
FROM	 schools s LEFT JOIN school_details sd
		 ON s.schoolID = sd.schoolID
GROUP BY s.schoolID
ORDER BY num_players DESC
LIMIT 	 5;

-- For each decade, what were the names of the top 3 schools that produced the most players? [Window Functions]
WITH ds AS (SELECT	 FLOOR(s.yearID / 10) * 10 AS decade, sd.name_full, COUNT(DISTINCT s.playerID) AS num_players
			FROM	 schools s LEFT JOIN school_details sd
					 ON s.schoolID = sd.schoolID
			GROUP BY decade, s.schoolID),
            
	 rn AS (SELECT	decade, name_full, num_players,
					DENSE_RANK() OVER (PARTITION BY decade ORDER BY num_players DESC) AS row_num
                  
			FROM	ds)
            
SELECT	decade, name_full, num_players
FROM	rn
WHERE	row_num <= 3
ORDER BY decade DESC, row_num;

--SALARY ANALYSIS

-- Return the top 20% of teams in terms of average annual spending [Window Functions]
WITH ts AS (SELECT 	teamID, yearID, SUM(salary) AS total_spend
			FROM	salaries
			GROUP BY teamID, yearID),
            
	 sp AS (SELECT	teamID, AVG(total_spend) AS avg_spend,
					NTILE(5) OVER (order by AVG(total_spend) desc) AS spend_pct
			FROM	ts
			GROUP BY teamID)
            
SELECT	teamID, ROUND(avg_spend / 1000000, 1) AS avg_spend_millions
FROM	sp
WHERE	spend_pct = 1;

-- For each team, show the cumulative sum of spending over the years [Rolling Calculations]
WITH ts AS (SELECT	 teamID, yearID, SUM(salary) AS total_spend
			FROM	 salaries
			GROUP BY teamID, yearID)
            
SELECT	teamID, yearID,
		ROUND(SUM(total_spend) OVER (PARTITION BY teamID ORDER BY yearID) / 1000000, 1)
			AS cumulative_sum_millions
FROM	ts;

-- Return the first year that each team's cumulative spending surpassed 1 billion 
WITH ts AS (SELECT	 teamID, yearID, SUM(salary) AS total_spend
			FROM	 salaries
			GROUP BY teamID, yearID),
            
	 cs AS (SELECT	teamID, yearID,
					SUM(total_spend) OVER (PARTITION BY teamID ORDER BY yearID)
						AS cumulative_sum
			FROM	ts),
            
	 rn AS (SELECT	teamID, yearID, cumulative_sum,
					ROW_NUMBER() OVER (PARTITION BY teamID ORDER BY cumulative_sum) AS rn
			FROM	cs
			WHERE	cumulative_sum > 1000000000)
            
SELECT	teamID, yearID, ROUND(cumulative_sum / 1000000000, 2) AS cumulative_sum_billions
FROM	rn
WHERE	rn = 1;

--- PLAYER CAREER ANALYSIS

-- For each player, calculate their age at their first (debut) game, their last game,
-- and their career length (all in years). Sort from longest career to shortest career. [Datetime Functions]
SELECT 	nameGiven,
        TIMESTAMPDIFF(YEAR, CAST(CONCAT(birthYear, '-', birthMonth, '-', birthDay) AS DATE), debut)
			AS starting_age,
		TIMESTAMPDIFF(YEAR, CAST(CONCAT(birthYear, '-', birthMonth, '-', birthDay) AS DATE), finalGame)
			AS ending_age,
		TIMESTAMPDIFF(YEAR, debut, finalGame) AS career_length
FROM	players
ORDER BY career_length DESC;

-- What team did each player play on for their starting and ending years? [Joins]
SELECT 	p.nameGiven,
		s.yearID AS starting_year, s.teamID AS starting_team,
        e.yearID AS ending_year, e.teamID AS ending_team
FROM	players p INNER JOIN salaries s
							ON p.playerID = s.playerID
							AND YEAR(p.debut) = s.yearID
				  INNER JOIN salaries e
							ON p.playerID = e.playerID
							AND YEAR(p.finalGame) = e.yearID;

--  How many players started and ended on the same team and also played for over a decade? [Basics]
SELECT 	p.nameGiven,
		s.yearID AS starting_year, s.teamID AS starting_team,
        e.yearID AS ending_year, e.teamID AS ending_team
FROM	players p INNER JOIN salaries s
							ON p.playerID = s.playerID
							AND YEAR(p.debut) = s.yearID
				  INNER JOIN salaries e
							ON p.playerID = e.playerID
							AND YEAR(p.finalGame) = e.yearID
WHERE	s.teamID = e.teamID AND e.yearID - s.yearID > 10;

-- PLAYER COMPARISON ANALYSIS

--Which players have the same birthday? Hint: Look into GROUP_CONCAT / LISTAGG / STRING_AGG [String Functions]
WITH bn AS (SELECT	CAST(CONCAT(birthYear, '-', birthMonth, '-', birthDay) AS DATE) AS birthdate,
					nameGiven
			FROM	players)
            
SELECT	birthdate, GROUP_CONCAT(nameGiven SEPARATOR ', ') AS players
FROM	bn
WHERE	YEAR(birthdate) BETWEEN 1980 AND 1990
GROUP BY birthdate
ORDER BY birthdate;

--  Create a summary table that shows for each team, what percent of players bat right, left and both [Pivoting]

-- first removes duplicate playerID-teamID combos from the salaries table before pivoting
-- The duplicates exist because each row in the salaries table represents a player's salary on a particular team for a particular year
WITH up AS (SELECT DISTINCT s.teamID, s.playerID, p.bats
           FROM salaries s LEFT JOIN players p
           ON s.playerID = p.playerID) 

SELECT teamID,
		ROUND(SUM(CASE WHEN bats = 'R' THEN 1 ELSE 0 END) / COUNT(playerID) * 100, 1) AS bats_right,
        ROUND(SUM(CASE WHEN bats = 'L' THEN 1 ELSE 0 END) / COUNT(playerID) * 100, 1) AS bats_left,
        ROUND(SUM(CASE WHEN bats = 'B' THEN 1 ELSE 0 END) / COUNT(playerID) * 100, 1) AS bats_both
FROM up
GROUP BY teamID;

-- How have average height and weight at debut game changed over the years, and what's the decade-over-decade difference? [Window Functions]
WITH hw AS (SELECT	FLOOR(YEAR(debut) / 10) * 10 AS decade,
					AVG(height) AS avg_height, AVG(weight) AS avg_weight
			FROM	players
			GROUP BY decade)
            
SELECT	decade,
		avg_height - LAG(avg_height) OVER(ORDER BY decade) AS height_diff,
        avg_weight - LAG(avg_weight) OVER(ORDER BY decade) AS weight_diff
FROM	hw
WHERE	decade IS NOT NULL;
