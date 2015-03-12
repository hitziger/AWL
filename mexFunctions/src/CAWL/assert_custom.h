
#ifndef ASSERT_CUSTOM_H
#define ASSERT_CUSTOM_H


#ifdef MEX_ASSERT
#include <mex.h>
void custom_assert(bool assertion, const char* msg) {
    if (!assertion) mexErrMsgTxt(msg);
}
void exit() {
    mexErrMsgTxt("Exiting program...");
}

#else
#include <iostream>
#include <cassert>
void custom_assert(bool assertion, const char* msg) {
    if (!assertion) std::cout << msg << std::endl;
    assert(assertion);
}
void exit() {
    assert(0);
}
#endif


#endif
