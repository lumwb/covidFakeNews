-- sqlite3 library.db < library-schema.sql

drop table if exists book;
create table book (
  id integer primary key autoincrement,
  author_id integer,
  title text not null
);
