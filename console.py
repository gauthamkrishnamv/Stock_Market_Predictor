#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QProcess, QTextCodec
from PyQt5.QtGui import QTextCursor, QKeySequence
from PyQt5.QtWidgets import QApplication, QPlainTextEdit, QShortcut


class ProcessOutputReader(QProcess):
    produce_output = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # merge stderr channel into stdout channel
        self.setProcessChannelMode(QProcess.MergedChannels)

        # prepare decoding process' output to Unicode
        codec = QTextCodec.codecForLocale()
        self._decoder_stdout = codec.makeDecoder()
        # only necessary when stderr channel isn't merged into stdout:
        # self._decoder_stderr = codec.makeDecoder()

        self.readyReadStandardOutput.connect(self._ready_read_standard_output)
        # only necessary when stderr channel isn't merged into stdout:
        # self.readyReadStandardError.connect(self._ready_read_standard_error)
        

    @pyqtSlot()
    def _ready_read_standard_output(self):
        raw_bytes = self.readAllStandardOutput()
        text = self._decoder_stdout.toUnicode(raw_bytes)
        self.produce_output.emit(text)

    # only necessary when stderr channel isn't merged into stdout:
    # @pyqtSlot()
    # def _ready_read_standard_error(self):
    #     raw_bytes = self.readAllStandardError()
    #     text = self._decoder_stderr.toUnicode(raw_bytes)
    #     self.produce_output.emit(text)


class Console(QPlainTextEdit):

    def __init__(self, reader, parent=None):
        super().__init__(parent=parent)
        self.reader = reader
        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)  # limit console to 10000 lines

        self._cursor_output = self.textCursor()
        self.kill_shortcut = QShortcut(QKeySequence('Ctrl+C'), self)
        self.kill_shortcut.activated.connect(self.kill_process)
        
    def kill_process(self):
        self.reader.kill()
        self.append_output('PROCESS KILLED!!')

    @pyqtSlot(str)
    def append_output(self, text):
        self._cursor_output.insertText(text)
        self.scroll_to_last_line()

    def scroll_to_last_line(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.Up if cursor.atBlockStart() else
                            QTextCursor.StartOfLine)
        self.setTextCursor(cursor)
    

    @pyqtSlot(str)
    def display(self, text):
        self.setPlainText(text)
        # print(text)

if __name__ == '__main__':
# create the application instance
    app = QApplication(sys.argv)

    # create a process output reader
    reader = ProcessOutputReader()

    # create a console and connect the process output reader to it
    console = Console(reader)
    reader.produce_output.connect(console.setPlainText)

    reader.start('python', ['-u', 'test_process.py'])  # start the process
    console.show()                              # make the console visible
    app.exec_()                                 # run the PyQt main loop