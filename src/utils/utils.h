
#ifndef UTILS_H
#define UTILS_H

namespace Utils {

	template<typename A, typename B>
std::pair<B,A> flip_pair(const std::pair<A,B> &p)
{
	return std::pair<B,A>(p.second, p.first);
}

	template<typename A, typename B>
std::map<B,A> flip_map(const std::map<A,B> &src)
{
	std::map<B,A> dst;
	std::transform(src.begin(), src.end(), std::inserter(dst, dst.begin()), 
			flip_pair<A,B>);
	return dst;
}

}


#endif /* end of include guard: UTILS_H */
