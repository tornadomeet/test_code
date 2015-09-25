syntax enable
:set nu
:colorscheme desert

set nocompatible                " no compatible with vi
filetype off                    " required!
set rtp+=~/.vim/bundle/vundle/
call vundle#rc()

set pastetoggle=<F9>   " paste mode and nopaset mode toggle

nmap ,cc <leader>cc
nmap ,cu <leader>cu
nmap ,cm <leader>cm

" let Vundle manage Vundle
Bundle 'gmarik/vundle'

"my Bundle here:
"
" original repos on github
"Bundle 'kien/ctrlp.vim'
"Bundle 'sukima/xmledit'
"Bundle 'sjl/gundo.vim'
"Bundle 'jiangmiao/auto-pairs'
"Bundle 'klen/python-mode'
"Bundle 'Valloric/ListToggle'
"Bundle 'SirVer/ultisnips'
"Bundle 'Valloric/YouCompleteMe'
"Bundle 'scrooloose/syntastic'
"Bundle 't9md/vim-quickhl'
"Bundle 'Lokaltog/vim-powerline'
"Bundle 'scrooloose/nerdcommenter'
"..................................
" vim-scripts repos
"Bundle 'YankRing.vim'
"Bundle 'vcscommand.vim'
"Bundle 'ShowPairs'
"Bundle 'SudoEdit.vim'
"Bundle 'EasyGrep'
"Bundle 'VOoM'
"Bundle 'VimIM'
"

" powerline
Bundle 'Lokaltog/vim-powerline'
set laststatus=2
set t_Co=256

" Auto complement the punctuation
Bundle 'AutoClose'

" Display the file catalog in the tree file
Bundle 'The-NERD-tree'

" find file using Ctrl+P
Bundle 'ctrlp.vim'

" comment the code, can support many language"
Bundle 'The-NERD-Commenter'

"..................................
" non github repos
" Bundle 'git://git.wincent.com/command-t.git'
"......................................
filetype plugin indent on
