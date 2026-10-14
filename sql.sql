show databases;
drop database blog;
create database if not exists blog;

show collation;

use blog;
show tables;

drop table  user;
create table if not exists user (
                                    id int primary key auto_increment,
                                    username varchar(256),
    account varchar(256) not null,
    ip varchar(256)
    );

desc user;

insert into user(id, username, account, ip) value (null,'1','1','1');


select * from user;

update user set username = '2' where id = 1;

delete from user where id = 1;

alter table user
    add email varchar(256) default '0' after ip,
    modify account char(25) not null ,
drop username,
    rename column id to uid;

