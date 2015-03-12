/*!
 *
 *                File utils.h
 *
 * Miscellaneous utility functions:
 * - min/max
 * - timer
 * - quick sort algorithm
 *
 * */

#ifndef UTILS_H
#define UTILS_H


// min and max
template<typename T>
T max(const T lhs, const T rhs) {
    return lhs>rhs? lhs : rhs;
}
template<typename T>
T min(const T lhs, const T rhs) {
    return lhs>rhs? rhs : lhs;
}


///////////////////////////////////////////////////////////////////////////////
////   CLASS TIMER                                                         ////
///////////////////////////////////////////////////////////////////////////////
////       -handles timing of processes                                    ////
///////////////////////////////////////////////////////////////////////////////
class timer {
private:
    unsigned long begTime;
    double accTime;
    bool running;
public:
    timer(): accTime(0.), running(false) {}

    void start() {
        custom_assert(!running,"timer::start: already running!");
        begTime = clock();
        running = true;
    }
    void stop() {
        custom_assert(running,"timer::stop: already stopped!");
        accTime += ((double) ((unsigned long) clock() - begTime))
            / CLOCKS_PER_SEC;
        running = false;
    }
    double elapsedTime() {
        return accTime;
    }
    bool isTimeout(double seconds) {
        return seconds >= elapsedTime();
    }
};

///////////////////////////////////////////////////////////////////////////////
////   SORTING ROUTINES QUICK_INCR, QUICK_DECR, QUICK_SORT                 ////
///////////////////////////////////////////////////////////////////////////////
////       -sort array in place                                            ////
///////////////////////////////////////////////////////////////////////////////

// sort first array, apply same order to second array (ascending)
template <typename T1, typename T2>
static void quick_incr(T1* irout, T2* prout,const int beg, const int end) {
    if (end-1 <= beg) return;
    int pivot=beg;
    const T2 val_pivot=prout[pivot];
    const T1 key_pivot=irout[pivot];
    for (int i = beg+1; i<end; ++i) {
        if (irout[i] < key_pivot) {
//        if (prout[i] < val_pivot) {
            prout[pivot]=prout[i];
            irout[pivot]=irout[i];
            prout[i]=prout[++pivot];
            irout[i]=irout[pivot];
        } 
    }
    prout[pivot]=val_pivot;
    irout[pivot]=key_pivot;
    quick_incr(irout,prout,beg,pivot);
    quick_incr(irout,prout,pivot+1,end);
}
// sort first array, apply same order to second array (descending)
template <typename T1, typename T2>
static void quick_decr(T1* irout, T2* prout,const int beg, const int end) {
    if (end-1 <= beg) return;
    int pivot=beg;
    const T2 val_pivot=prout[pivot];
    const T1 key_pivot=irout[pivot];
    for (int i = beg+1; i<end; ++i) {
        if (irout[i] > key_pivot) {
//        if (prout[i] > val_pivot) {
            prout[pivot]=prout[i];
            irout[pivot]=irout[i];
            prout[i]=prout[++pivot];
            irout[i]=irout[pivot];
        } 
    }
    prout[pivot]=val_pivot;
    irout[pivot]=key_pivot;
    quick_decr(irout,prout,beg,pivot);
    quick_decr(irout,prout,pivot+1,end);
}

// sort first vector, apply same order to second array (ascending)
template <typename T>
static void quick_incr(std::vector<int>& irOut, std::vector<T>& prOut,
        const int beg, const int end) {
    if (end-1 <= beg) return;
    int pivot=beg;
    const T val_pivot=prOut[pivot];
    const int key_pivot=irOut[pivot];
    for (int i = beg+1; i<end; ++i) {
        if (irOut[i] < key_pivot) {
//        if (prout[i] < val_pivot) {
            prOut[pivot]=prOut[i];
            irOut[pivot]=irOut[i];
            prOut[i]=prOut[++pivot];
            irOut[i]=irOut[pivot];
        } 
    }
    prOut[pivot]=val_pivot;
    irOut[pivot]=key_pivot;
    quick_incr(irOut,prOut,beg,pivot);
    quick_incr(irOut,prOut,pivot+1,end);
}
// sort first vector, apply same order to second array (descending)
template <typename T>
static void quick_decr(std::vector<int>& irOut, std::vector<T>& prOut,
        const int beg, const int end) {
    if (end-1 <= beg) return;
    int pivot=beg;
    const T val_pivot=prOut[pivot];
    const int key_pivot=irOut[pivot];
    for (int i = beg+1; i<end; ++i) {
        if (irOut[i] > key_pivot) {
//        if (prOut[i] > val_pivot) {
            prOut[pivot]=prOut[i];
            irOut[pivot]=irOut[i];
            prOut[i]=prOut[++pivot];
            irOut[i]=irOut[pivot];
        } 
    }
    prOut[pivot]=val_pivot;
    irOut[pivot]=key_pivot;
    quick_decr(irOut,prOut,beg,pivot);
    quick_decr(irOut,prOut,pivot+1,end);
}

// sort first vector, apply same order to second array (ascending for incr=true,
// otherwise descending)
template <typename T>
static void quick_sort(std::vector<int>& In, std::vector<T>& Out,
        const bool incr=true) {
    custom_assert(In.size()==Out.size(),
            "quick_sort: vectors must be of same length");
    if (incr) quick_incr(In,Out,int(),In.size());
    else      quick_decr(In,Out,int(),In.size());
}

#endif
