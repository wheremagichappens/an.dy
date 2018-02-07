create database movies;

use movies;



DROP TABLE IF EXISTS movie_rating;
DROP TABLE IF EXISTS movie_watchers;
DROP TABLE IF EXISTS movie_title;


CREATE TABLE movie_rating (
user_id integer NOT NULL,
movie_id integer NOT NULL,
rating integer NOT NULL
);


LOAD DATA LOCAL INFILE 'c:/data/movies/movie_rating.csv' 
INTO TABLE movie_rating 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;



CREATE TABLE movie_watchers (
user_id integer NOT NULL,
name varchar(255) NOT NULL
);


LOAD DATA LOCAL INFILE 'c:/data/movies/movie_watchers.csv' 
INTO TABLE movie_watchers 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;



CREATE TABLE movie_title (
movie_id integer NOT NULL,
title varchar(255) NOT NULL
);


LOAD DATA LOCAL INFILE 'c:/data/movies/movie_title.csv' 
INTO TABLE movie_title 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

select * from movie_rating;
select * from movie_watchers;
select * from movie_title;


select b.name, c.title, a.rating
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/Uploads/rating4.csv'
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
ESCAPED BY '\\'
LINES TERMINATED BY '\n'
from movie_rating a
join movie_watchers b on a.user_id = b.user_id
join movie_title c on a.movie_id = c.movie_id
order by b.name;