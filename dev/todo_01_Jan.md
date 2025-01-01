- [x] add support for collection names in the code (use env settings attrs instead of str consts)
- [ ] do a dynamic get_messages(chat) and get_chats() that check for updates in addition to using the db

- [ ] test full pipeline
- [x] use ignore_finished flag

- [ ] add access utils to the telegram_downloader class
  - utils: find_chat
  - utils: get_messages
- [ ] load messages in a jupyter notebook
    - show messages / format messages or somethign
	- show chat names


- [ ] figure out logging. 1) I want minimalistic logging by default. 2) I want to have cool info 3) debug logging
  - [ ] new flag: --silent
  - [ ] use info for all cool info
  - [ ] report the messages that are already in the db
  - [ ] use debug for ALL debug info and steps
  - [ ] save chat id to messages, I guess - check if I can filter 

- [ ] sanity check message duplicates (only complete duplicates - allow changed edits history)
- [ ] add media downloading -> aws s3


- [ ] fix the correct way to find existing messages for the chat
  - [ ] try in the notebook
- [ ] add sanity check: if timestamp < last message date for chat -> skip